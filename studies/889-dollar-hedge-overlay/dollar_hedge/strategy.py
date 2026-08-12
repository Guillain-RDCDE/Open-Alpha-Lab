"""Strategy + inference for Study 889 — Broad Dollar-Hedge Overlay.

The mechanical identity under test (log-approx, per month), generalised from 613's single Japan
pair to broad developed international (MSCI EAFE):

    diff := hedged - unhedged  =  carry - fx_foreign  =  carry + dollar_return
    =>  carry_hat := diff + fx_foreign = diff - dollar_return

where ``dollar_return`` is the broad-dollar (UUP) return and ``carry = (r_US - r_foreign)/12``. If
the hedge really pockets the differential:

  1. ``carry_hat`` has mean ~ (r_US - r_foreign)/12 with a HAC t that clears 2 in the era the US
     out-yields (2022+);
  2. regressing ``diff`` on ``dollar_return`` gives beta ~ 1 (the hedge is a full long of the
     dollar / short of the foreign basket) and alpha ~ the mean carry (plus basket-alpha for the
     mismatched pair);
  3. the excess-of-cash Sharpe of the hedged sleeve beats the unhedged sleeve *when* the
     differential is positive — and a "hedge when the US out-yields" overlay tries to harvest that.

Inference: Newey-West (HAC) t on means and on OLS coefficients (monthly hedge-roll timing induces
serial correlation, so plain t's are banned). One documented execution lag: the overlay uses the
PRIOR month-end differential to pick the class held over the NEXT month. Costs: one-way bps x NAV
on each switch; the long-hedged/short-unhedged isolation spread pays borrow on the short leg.
Excess-of-cash Sharpes subtract the BIL monthly return from both legs (excess-vs-excess race).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12
NW_LAGS = 6  # Newey-West lag window for monthly series (~ 1.5 * T^(1/3) at T ~ 150)


# --------------------------------------------------------------------------- #
# HAC inference primitives (Newey-West / Bartlett)
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


def newey_west_t(x: np.ndarray, lags: int = NW_LAGS) -> float:
    """HAC t of mean(x) vs 0 (convenience wrapper)."""
    return nw_mean_t(x, lags)[1]


def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def hac_ols(y: np.ndarray, x: np.ndarray, lags: int = NW_LAGS) -> dict:
    """OLS of y on [1, x] with Newey-West HAC standard errors.

    Returns alpha, beta, their HAC t's, and R^2. Used for ``diff = alpha + beta * dollar + e``.
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
    Z = X * e[:, None]
    S = Z.T @ Z
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
# Sharpe helpers (annualised from monthly, excess-of-cash aware)
# --------------------------------------------------------------------------- #
def ann_sharpe(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(MONTHS)) if sd > 0 else float("nan")


def sharpe_boot_ci(r: np.ndarray, n_boot: int = 2000, block: int = 3,
                   seed: int = 889, alpha: float = 0.05) -> dict:
    """Circular block-bootstrap CI for the annualised Sharpe of a monthly series."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    rng = np.random.default_rng(seed)
    blk = max(1, min(block, n))
    n_blocks = int(np.ceil(n / blk))
    offs = np.arange(blk)
    boots = np.full(n_boot, np.nan)
    ann = np.sqrt(MONTHS)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offs[None, :]) % n).ravel()[:n]
        s = r[idx]
        sd = s.std(ddof=1)
        if sd > 0:
            boots[b] = s.mean() / sd * ann
    v = boots[np.isfinite(boots)]
    lo, hi = np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"sharpe": ann_sharpe(r), "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((v < 0).mean()), "n_boot_valid": int(v.size)}


def mean_boot_ci(x: np.ndarray, n_boot: int = 2000, block: int = 3,
                 seed: int = 889, alpha: float = 0.05) -> dict:
    """Circular block-bootstrap CI for the annualised mean (%/yr) of a monthly series."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    rng = np.random.default_rng(seed)
    blk = max(1, min(block, n))
    n_blocks = int(np.ceil(n / blk))
    offs = np.arange(blk)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offs[None, :]) % n).ravel()[:n]
        boots[b] = x[idx].mean() * MONTHS * 100
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean_ann_pct": float(x.mean() * MONTHS * 100),
            "ci_low": float(lo), "ci_high": float(hi),
            "frac_le_zero": float((boots <= 0).mean())}


# --------------------------------------------------------------------------- #
# The carry decomposition per pair
# --------------------------------------------------------------------------- #
def pair_frame(panel: pd.DataFrame, hedged: str, unhedged: str,
               start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Aligned monthly frame for one hedged/unhedged pair.

    ``diff`` = hedged - unhedged; ``fx_foreign`` = SPOT USD return of the EAFE currency basket;
    ``carry_hat`` = diff + fx_foreign (the hedge-P&L carry estimate); ``dollar`` = tradeable-dollar
    (UUP) return kept for the collateral-yield contrast; ``diff_rate`` = observable US-minus-foreign
    policy differential (annual %); ``h_ex`` / ``u_ex`` = excess-of-cash leg returns (minus BIL).
    """
    df = pd.DataFrame({
        "hedged": panel[hedged], "unhedged": panel[unhedged],
        "fx_foreign": panel["fx_foreign"], "dollar": panel["dollar"],
        "cash": panel["BIL"], "diff_rate": panel["diff_rate"],
    }).dropna(subset=["hedged", "unhedged", "fx_foreign"])
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    df["diff"] = df["hedged"] - df["unhedged"]
    df["carry_hat"] = df["diff"] + df["fx_foreign"]
    df["h_ex"] = df["hedged"] - df["cash"]
    df["u_ex"] = df["unhedged"] - df["cash"]
    return df


def pair_stats(pf: pd.DataFrame, lags: int = NW_LAGS) -> dict:
    """Headline stats for one pair window: carry mean/HAC t, hedge regression, excess Sharpes."""
    mean_diff, t_diff = nw_mean_t(pf["diff"].values, lags)
    mean_carry, t_carry = nw_mean_t(pf["carry_hat"].values, lags)
    # the hedge is a full SHORT of the foreign basket: diff = alpha + beta * (-fx_foreign)
    reg = hac_ols(pf["diff"].values, -pf["fx_foreign"].values, lags)
    sh_h = ann_sharpe(pf["h_ex"].values)
    sh_u = ann_sharpe(pf["u_ex"].values)
    return {
        "n": len(pf),
        "start": str(pf.index.min().date()), "end": str(pf.index.max().date()),
        "diff_ann_pct": mean_diff * MONTHS * 100, "t_diff": t_diff,
        "carry_ann_pct": mean_carry * MONTHS * 100, "t_carry": t_carry,
        "rate_diff_ann_pct": float(pf["diff_rate"].mean()),
        "alpha_ann_pct": reg["alpha"] * MONTHS * 100, "t_alpha": reg["t_alpha"],
        "beta": reg["beta"], "t_beta": reg["t_beta"], "r2": reg["r2"],
        "sharpe_hedged_ex": sh_h, "sharpe_unhedged_ex": sh_u,
        "sharpe_adv": sh_h - sh_u,
    }


def era_split(pf: pd.DataFrame, cut: str = "2022-01-01", lags: int = NW_LAGS) -> dict:
    """Carry mean + HAC t in the pre- and post-cut eras (default split at the 2022 dollar regime)."""
    out = {}
    for lbl, sub in [("pre", pf[pf.index < pd.Timestamp(cut)]),
                     ("post", pf[pf.index >= pd.Timestamp(cut)])]:
        if len(sub) >= 8:
            mc, tc = nw_mean_t(sub["carry_hat"].values, lags)
            out[lbl] = {"n": len(sub), "carry_ann_pct": mc * MONTHS * 100, "t_carry": tc,
                        "rate_diff_ann_pct": float(sub["diff_rate"].mean())}
        else:
            out[lbl] = {"n": len(sub), "carry_ann_pct": float("nan"),
                        "t_carry": float("nan"), "rate_diff_ann_pct": float("nan")}
    return out


def calendar_years(pf: pd.DataFrame) -> pd.DataFrame:
    """Per-calendar-year compounded hedged / unhedged / diff returns (%)."""
    g = pf.groupby(pf.index.year)
    comp = lambda s: (1.0 + s).prod() - 1.0
    tab = pd.DataFrame({
        "hedged_%": g["hedged"].apply(comp) * 100,
        "unhedged_%": g["unhedged"].apply(comp) * 100,
    })
    tab["diff_%"] = tab["hedged_%"] - tab["unhedged_%"]
    tab["dollar_%"] = g["dollar"].apply(comp) * 100
    return tab


def max_drawdown(r: np.ndarray) -> float:
    """Max drawdown (fraction, negative) of a compounded monthly return series."""
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return float("nan")
    curve = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(curve)
    return float((curve / peak - 1.0).min())


# --------------------------------------------------------------------------- #
# The "hedge when the US out-yields" overlay
# --------------------------------------------------------------------------- #
def overlay_switch(pf: pd.DataFrame, thresh_ann_pct: float = 0.0,
                   cost_bps_oneway: float = 3.0, lags: int = NW_LAGS) -> dict:
    """Hold the HEDGED class when the PRIOR month-end US-minus-foreign differential > thresh,
    else the unhedged class. ONE execution lag: the signal at month-end t-1 picks the class held
    over month t (the differential is observable policy rates, genuinely knowable). A full switch
    costs 2 x one-way bps (sell one class, buy the other). Reports the overlay's excess-of-cash
    Sharpe against the two always-on benchmarks."""
    sig = (pf["diff_rate"].shift(1) > thresh_ann_pct)
    sig.iloc[0] = bool(pf["diff_rate"].iloc[0] > thresh_ann_pct)
    strat = np.where(sig, pf["hedged"].values, pf["unhedged"].values)
    switches = int((sig != sig.shift(1)).iloc[1:].sum())
    cost_total = switches * 2 * cost_bps_oneway / 1e4
    n = len(pf)
    strat_net = strat.copy()
    # amortise the (few) switch costs across the sample as a flat monthly drag
    strat_net = strat_net - cost_total / n
    ex = strat_net - pf["cash"].values
    h_ex = pf["h_ex"].values
    u_ex = pf["u_ex"].values
    return {
        "n": n, "switches": switches, "share_hedged": float(sig.mean()),
        "overlay_ann_pct": float(np.nanmean(strat_net)) * MONTHS * 100,
        "sharpe_overlay_ex": ann_sharpe(ex),
        "sharpe_hedged_ex": ann_sharpe(h_ex),
        "sharpe_unhedged_ex": ann_sharpe(u_ex),
        "adv_vs_unhedged": ann_sharpe(ex) - ann_sharpe(u_ex),
        "adv_vs_hedged": ann_sharpe(ex) - ann_sharpe(h_ex),
        "cost_drag_ann_pct": cost_total / n * MONTHS * 100,
    }


def spread_trade(pf: pd.DataFrame, borrow_annual_bps: float = 50.0,
                 cost_bps_oneway: float = 3.0, turnover_per_year: float = 2.0,
                 lags: int = NW_LAGS) -> dict:
    """Isolate the carry: long hedged / short unhedged (dollar-neutral, per $ of long NAV).

    Gross monthly P&L = diff. Charges: borrow on the short leg (annual bps / 12) and one-way costs
    x NAV on ~turnover_per_year rebalances of BOTH legs per year. This spread is long the dollar
    (short the foreign basket); ``net_carry`` strips the dollar leg and reports the fx-neutral
    carry after the same charges."""
    charge_m = borrow_annual_bps / 1e4 / MONTHS + 2 * cost_bps_oneway / 1e4 * turnover_per_year / MONTHS
    net = pf["diff"].values - charge_m
    net_carry = pf["carry_hat"].values - charge_m
    m, t = nw_mean_t(net, lags)
    mc, tc = nw_mean_t(net_carry, lags)
    return {"net_diff_ann_pct": m * MONTHS * 100, "t_net_diff": t,
            "net_carry_ann_pct": mc * MONTHS * 100, "t_net_carry": tc,
            "charge_ann_pct": charge_m * MONTHS * 100}


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(world: pd.DataFrame, lags: int = NW_LAGS) -> dict:
    """Run the carry estimator on a synthetic world; recover the planted carry."""
    pf = pair_frame(world, "HEFA", "EFA")
    st = pair_stats(pf, lags)
    return {"carry_ann_pct": st["carry_ann_pct"], "t_carry": st["t_carry"],
            "beta": st["beta"], "t_beta": st["t_beta"], "n": st["n"]}
