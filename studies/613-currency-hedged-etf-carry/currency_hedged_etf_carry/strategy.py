"""Strategy + inference for Study 613 — Currency-Hedged ETF Carry.

The mechanical identity under test (log-approx, per month):

    hedged - unhedged  =  carry - fx      =>      carry_hat := (hedged - unhedged) + fx

where ``fx`` is the USD return of holding the foreign currency. If the hedge really pockets the
short-rate differential, then

  1. ``carry_hat`` has mean ~ (r_US - r_foreign)/12 with a HAC t that clears 2 when the
     differential is large (the 2022+ era: ~5 %/yr vs JPY);
  2. regressing (hedged - unhedged) on ``-fx`` gives beta ~ 1 (the hedge is a full short of the
     currency) and alpha ~ the mean carry (plus any basket-alpha for mismatched pairs);
  3. the 12-month rolling ``carry_hat`` tracks the *observable* policy-rate differential.

Inference: Newey-West (HAC) t on means and on OLS coefficients — monthly return differentials
are serially correlated through the hedge-roll timing, so plain t's are banned here. One
execution lag, documented: the share-class switch strategy uses the PRIOR month-end rate
differential to choose the class held over the NEXT month. Costs: one-way bps x NAV on each
switch; the long-hedged/short-unhedged isolation trade pays borrow on the short leg.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NW_LAGS = 6  # Newey-West lag window for monthly series (~ 1.5 * T^(1/3) at T ~ 200)


# --------------------------------------------------------------------------- #
# HAC inference primitives
# --------------------------------------------------------------------------- #
def nw_mean_t(x: np.ndarray, lags: int = NW_LAGS) -> tuple[float, float]:
    """Mean of ``x`` and its Newey-West (Bartlett) HAC t-statistic."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 8:
        return float("nan"), float("nan")
    e = x - x.mean()
    g0 = float(e @ e) / n
    s = g0
    for k in range(1, min(lags, n - 1) + 1):
        gk = float(e[k:] @ e[:-k]) / n
        s += 2.0 * (1.0 - k / (lags + 1.0)) * gk
    se = np.sqrt(max(s, 1e-18) / n)
    return float(x.mean()), float(x.mean() / se)


def hac_ols(y: np.ndarray, x: np.ndarray, lags: int = NW_LAGS) -> dict:
    """OLS of y on [1, x] with Newey-West HAC standard errors.

    Returns alpha, beta, their HAC t's, and R^2. Used for
    (hedged - unhedged)_t = alpha + beta * (-fx_t) + e_t.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    ok = ~(np.isnan(y) | np.isnan(x))
    y, x = y[ok], x[ok]
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XtX_inv @ (X.T @ y)
    e = y - X @ beta_hat
    # Newey-West covariance of the score
    S = np.zeros((2, 2))
    Z = X * e[:, None]
    S += Z.T @ Z
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        G = Z[k:].T @ Z[:-k]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    r2 = 1.0 - float(e @ e) / float(((y - y.mean()) ** 2).sum())
    return {
        "alpha": float(beta_hat[0]), "beta": float(beta_hat[1]),
        "t_alpha": float(beta_hat[0] / se[0]), "t_beta": float(beta_hat[1] / se[1]),
        "r2": r2, "n": n,
    }


# --------------------------------------------------------------------------- #
# The carry decomposition per pair
# --------------------------------------------------------------------------- #
def pair_frame(panel: pd.DataFrame, hedged: str, unhedged: str, ccy: str,
               start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Aligned monthly frame for one share-class pair: diff, fx, carry_hat, rate_diff.

    ``diff`` = hedged - unhedged monthly return; ``fx`` = USD return of the foreign currency;
    ``carry_hat`` = diff + fx (the hedge-P&L carry estimate); ``rate_diff`` = the observable
    policy-rate differential (annual %).
    """
    df = pd.DataFrame({
        "diff": panel[hedged] - panel[unhedged],
        "fx": panel[f"fx_{ccy}"],
        "rate_diff": panel[f"diff_{ccy}"],
    }).dropna()
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    df["carry_hat"] = df["diff"] + df["fx"]
    return df


def pair_stats(pf: pd.DataFrame, lags: int = NW_LAGS) -> dict:
    """Headline stats for one pair window: means (bps/mo), HAC t's, hedge regression."""
    mean_diff, t_diff = nw_mean_t(pf["diff"].values, lags)
    mean_carry, t_carry = nw_mean_t(pf["carry_hat"].values, lags)
    mean_fx, _ = nw_mean_t(pf["fx"].values, lags)
    reg = hac_ols(pf["diff"].values, -pf["fx"].values, lags)
    return {
        "n": len(pf),
        "start": str(pf.index.min().date()), "end": str(pf.index.max().date()),
        "diff_bps": mean_diff * 1e4, "t_diff": t_diff,
        "fx_bps": mean_fx * 1e4,
        "carry_bps": mean_carry * 1e4, "t_carry": t_carry,
        "carry_ann_pct": mean_carry * 12 * 100,
        "rate_diff_ann_pct": float(pf["rate_diff"].mean()),
        "alpha_bps": reg["alpha"] * 1e4, "t_alpha": reg["t_alpha"],
        "beta": reg["beta"], "t_beta": reg["t_beta"], "r2": reg["r2"],
    }


def rolling_carry_track(pf: pd.DataFrame, window: int = 12) -> dict:
    """12m rolling annualized carry_hat vs the rolling mean rate differential.

    Returns the two aligned series plus their correlation and the OLS slope of
    rolling carry on rolling differential (slope ~ 1 = the hedge passes the observable
    differential through, one-for-one, as it moves).
    """
    roll_c = pf["carry_hat"].rolling(window).mean() * 12 * 100   # annual %
    roll_d = pf["rate_diff"].rolling(window).mean()              # annual %
    ok = ~(roll_c.isna() | roll_d.isna())
    c, d = roll_c[ok], roll_d[ok]
    corr = float(np.corrcoef(c, d)[0, 1])
    slope = float(np.polyfit(d, c, 1)[0])
    return {"roll_carry": roll_c, "roll_diff": roll_d, "corr": corr,
            "slope": slope, "n": int(ok.sum())}


# --------------------------------------------------------------------------- #
# Tradability
# --------------------------------------------------------------------------- #
def spread_trade(pf: pd.DataFrame, borrow_annual_bps: float = 50.0,
                 cost_bps_oneway: float = 5.0, turnover_per_year: float = 2.0,
                 lags: int = NW_LAGS) -> dict:
    """Isolate the carry: long hedged / short unhedged (dollar-neutral, per $ of long NAV).

    Gross monthly P&L = diff. Charges: borrow on the short leg (annual bps / 12) and one-way
    costs x NAV on ~turnover_per_year rebalances of BOTH legs per year. Note the spread still
    carries the -fx leg (it is short the foreign currency), so this is the carry PLUS a
    currency bet unless fx ~ 0 on the window; ``carry_net`` reports the fx-stripped carry
    after the same charges.
    """
    charge_m = borrow_annual_bps / 1e4 / 12 + 2 * cost_bps_oneway / 1e4 * turnover_per_year / 12
    net = pf["diff"] - charge_m
    net_carry = pf["carry_hat"] - charge_m
    m, t = nw_mean_t(net.values, lags)
    mc, tc = nw_mean_t(net_carry.values, lags)
    return {"net_diff_ann_pct": m * 12 * 100, "t_net_diff": t,
            "net_carry_ann_pct": mc * 12 * 100, "t_net_carry": tc,
            "charge_ann_pct": charge_m * 12 * 100}


def switch_strategy(pf: pd.DataFrame, panel_pair: tuple[pd.Series, pd.Series],
                    thresh_ann_pct: float = 1.0, cost_bps_oneway: float = 5.0) -> dict:
    """Hold the HEDGED class when the PRIOR month-end rate differential > thresh, else the
    unhedged class. ONE execution lag: the signal at month-end t-1 picks the class held over
    month t (the differential is observable — policy rates — so this is genuinely knowable).
    A full switch costs 2 x one-way bps (sell one class, buy the other).
    """
    hedged_r, unhedged_r = panel_pair
    hr = hedged_r.reindex(pf.index)
    ur = unhedged_r.reindex(pf.index)
    sig = (pf["rate_diff"].shift(1) > thresh_ann_pct)          # decided at t-1, applied at t
    sig.iloc[0] = bool(pf["rate_diff"].iloc[0] > thresh_ann_pct)
    strat = np.where(sig, hr, ur)
    switches = int((sig != sig.shift(1)).iloc[1:].sum())
    cost_total = switches * 2 * cost_bps_oneway / 1e4
    n = len(pf)
    gross_ann = float(np.nanmean(strat)) * 12 * 100
    net_ann = gross_ann - cost_total / (n / 12) * 100
    return {
        "n": n, "switches": switches,
        "share_hedged": float(sig.mean()),
        "strat_net_ann_pct": net_ann,
        "hedged_ann_pct": float(hr.mean()) * 12 * 100,
        "unhedged_ann_pct": float(ur.mean()) * 12 * 100,
    }
