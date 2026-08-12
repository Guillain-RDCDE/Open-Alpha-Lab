"""Strategy + inference for Study 910 — Managed-Distribution CEF.

The buyer's question, stripped to arithmetic. Hold a persistent-discount, big-distribution CEF
(or the CEF-of-CEFs basket); measure everything **excess of cash** (minus BIL) so leverage can't
hide in the number:

    r_ex(fund)_t = r(fund)_t - r(BIL)_t

Then ask three things a sceptic asks of any "structural carry":

  1. **Is there an excess-of-cash return at all, and is it risk-adjusted-better than the asset
     class?** — the excess-vs-excess Sharpe *race*: annualised Sharpe of ``r_ex(fund)`` vs
     ``r_ex(SPY)``, plus the HAC *t* on the monthly mean of ``r_ex(fund)`` and on
     ``r(fund) - r(SPY)`` (does it beat the equity benchmark it is sold against?).
  2. **Or is it just levered beta?** — regress ``r_ex(fund)`` on ``r_ex(SPY)`` with Newey-West
     errors: ``beta`` (leverage/asset-class exposure) and ``alpha`` (the structural pickup that
     survives beta). A fat distribution that is all return-of-capital shows up as ``alpha ~ 0``
     (or negative) with ``beta`` doing all the work.
  3. **Does it survive stress and costs?** — a bootstrap Sharpe CI, the max drawdown and the
     calendar-year table, an era cut at the 2022 rate-hike regime (when leverage got expensive
     and bond CEFs' discounts blew out), and a costed net (CEF bid-ask × turnover per rebalance).

Inference is Newey-West HAC throughout — monthly CEF returns are serially correlated through
discount mean-reversion and leverage-roll timing, so plain t's are banned. One documented
rebalance lag on the equal-weight basket. The leverage financing cost is **already embedded** in
the total return (it is paid inside the fund), so no extra borrow charge is levied on the buyer —
the only trading friction is the CEF spread on the monthly rebalance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12
NW_LAGS = 6  # Newey-West lag window for monthly series (~ 1.5 * T^(1/3) at T ~ 150)


# --------------------------------------------------------------------------- #
# HAC inference primitives
# --------------------------------------------------------------------------- #
def nw_mean_t(x: np.ndarray, lags: int = NW_LAGS) -> tuple[float, float]:
    """Mean of ``x`` and its Newey-West (Bartlett) HAC t-statistic vs zero."""
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

    Returns alpha, beta, their HAC t's, and R². Used for the excess-vs-excess CAPM
    ``r_ex(fund)_t = alpha + beta * r_ex(SPY)_t + e_t``: ``beta`` is the leverage / asset-class
    exposure, ``alpha`` the structural pickup that survives beta.
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
# Return-series builders
# --------------------------------------------------------------------------- #
def equal_weight_basket(panel: pd.DataFrame, members: list[str]) -> pd.Series:
    """Monthly return of an equal-weight, monthly-rebalanced basket of ``members``.

    Each month's basket return is the simple mean of the members that have a return that month
    (the youngest name limits the front of the window). Monthly rebalancing to equal weight is
    the implicit one-lag rule: last month's weights are reset to 1/N at each month-end and held
    through the next month; the return recorded on month ``t`` uses only month-``t`` prices.
    """
    cols = [m for m in members if m in panel.columns]
    sub = panel[cols]
    # require ALL members present so the basket composition is constant (drops the pre-BST era)
    full = sub.dropna(how="any")
    return full.mean(axis=1)


def excess(series: pd.Series, cash: pd.Series) -> pd.Series:
    """Excess-of-cash monthly return, aligned on the intersection of the two indices."""
    df = pd.concat([series.rename("r"), cash.rename("c")], axis=1, sort=True).dropna()
    return (df["r"] - df["c"]).rename(series.name)


def ann_sharpe(x: np.ndarray) -> float:
    """Annualised Sharpe of a monthly *excess-of-cash* series (mean/sd * sqrt(12))."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(MONTHS)) if sd > 0 else float("nan")


def max_drawdown(monthly_ret: pd.Series) -> float:
    """Maximum drawdown of the cumulative (1+r) wealth curve (a negative fraction)."""
    r = monthly_ret.dropna().to_numpy(dtype=float)
    if len(r) == 0:
        return float("nan")
    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(wealth)
    return float((wealth / peak - 1.0).min())


def calendar_year_table(monthly_ret: pd.Series) -> pd.DataFrame:
    """Calendar-year compounded returns (%) of a monthly return series."""
    r = monthly_ret.dropna()
    grp = (1.0 + r).groupby(r.index.year).prod() - 1.0
    return (grp * 100).rename("ret_pct").to_frame()


# --------------------------------------------------------------------------- #
# The headline metrics for one fund vs the SPY benchmark and cash
# --------------------------------------------------------------------------- #
def fund_stats(fund_ret: pd.Series, spy_ret: pd.Series, cash_ret: pd.Series,
               lags: int = NW_LAGS) -> dict:
    """Excess-of-cash metrics for one fund: Sharpe race vs SPY, HAC t's, CAPM beta/alpha.

    All legs are aligned on the common (fund ∩ SPY ∩ cash) index so every comparison is on the
    same months. ``sharpe_adv`` = fund excess-of-cash Sharpe minus SPY excess-of-cash Sharpe.
    """
    df = pd.concat(
        [fund_ret.rename("f"), spy_ret.rename("s"), cash_ret.rename("c")], axis=1, sort=True
    ).dropna()
    f_ex = (df["f"] - df["c"]).to_numpy()
    s_ex = (df["s"] - df["c"]).to_numpy()
    vs_spy = (df["f"] - df["s"]).to_numpy()

    mean_ex, t_ex = nw_mean_t(f_ex, lags)
    mean_vs, t_vs = nw_mean_t(vs_spy, lags)
    reg = hac_ols(f_ex, s_ex, lags)

    f_sharpe = ann_sharpe(f_ex)
    s_sharpe = ann_sharpe(s_ex)
    return {
        "n": int(len(df)),
        "start": str(df.index.min().date()), "end": str(df.index.max().date()),
        "fund_ann_pct": float((1 + df["f"]).prod() ** (MONTHS / len(df)) - 1) * 100,
        "spy_ann_pct": float((1 + df["s"]).prod() ** (MONTHS / len(df)) - 1) * 100,
        "fund_exret_bps": mean_ex * 1e4, "t_exret": t_ex,
        "fund_sharpe": f_sharpe, "spy_sharpe": s_sharpe,
        "sharpe_adv": f_sharpe - s_sharpe,
        "vs_spy_bps": mean_vs * 1e4, "t_vs_spy": t_vs,
        "alpha_ann_pct": reg["alpha"] * MONTHS * 100, "t_alpha": reg["t_alpha"],
        "beta": reg["beta"], "t_beta": reg["t_beta"], "r2": reg["r2"],
        "max_dd_pct": max_drawdown(df["f"]) * 100,
        "fund_vol_pct": float(df["f"].std(ddof=1) * np.sqrt(MONTHS) * 100),
    }


# --------------------------------------------------------------------------- #
# Bootstrap Sharpe CI (moving-block, deterministic)
# --------------------------------------------------------------------------- #
def bootstrap_sharpe_ci(x: np.ndarray, n_boot: int = 5000, block: int = 6,
                        seed: int = 910, alpha: float = 0.05) -> dict:
    """Moving-block bootstrap CI for the annualised Sharpe of a monthly excess series.

    Blocks of ``block`` consecutive months preserve the serial dependence (discount
    mean-reversion). Returns the point Sharpe, the (1-alpha) percentile interval, and the share
    of resamples with a negative Sharpe (a blunt "could this be zero?" read).
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < block + 2:
        return {"sharpe": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "frac_neg": float("nan")}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block + 1
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, starts_max, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        s = x[idx]
        sd = s.std(ddof=1)
        boots[b] = s.mean() / sd * np.sqrt(MONTHS) if sd > 0 else np.nan
    boots = boots[np.isfinite(boots)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "sharpe": ann_sharpe(x), "ci_low": float(lo), "ci_high": float(hi),
        "frac_neg": float((boots < 0).mean()), "n_boot": int(len(boots)),
    }


# --------------------------------------------------------------------------- #
# Era cut
# --------------------------------------------------------------------------- #
def era_stats(fund_ret: pd.Series, spy_ret: pd.Series, cash_ret: pd.Series,
              split: str = "2022-01-01", lags: int = NW_LAGS) -> dict:
    """Split into pre-/post-``split`` (the 2022 rate-hike regime) and stat each half."""
    out = {}
    for lbl, lo, hi in [("pre", None, split), ("post", split, None)]:
        f = fund_ret.copy()
        if lo is not None:
            f = f[f.index >= pd.Timestamp(lo)]
        if hi is not None:
            f = f[f.index < pd.Timestamp(hi)]
        if len(f.dropna()) >= 10:
            out[lbl] = fund_stats(f, spy_ret, cash_ret, lags)
        else:
            out[lbl] = None
    return out


# --------------------------------------------------------------------------- #
# Costed net — the CEF bid-ask on the monthly rebalance
# --------------------------------------------------------------------------- #
def costed_net(basket_members_panel: pd.DataFrame, members: list[str],
               cash_ret: pd.Series, spy_ret: pd.Series,
               cost_bps_oneway: float = 15.0, lags: int = NW_LAGS) -> dict:
    """Charge the CEF spread on the equal-weight basket's monthly rebalance turnover.

    Each month the basket is reset to equal weight; turnover is the L1 change in weights (drift
    back to 1/N) plus nothing else on a pure hold — a conservative flat proxy is 2 × the mean
    absolute deviation of last month's realised member weights from 1/N. To stay comparable to
    the desk's other allocation timers we charge a flat ``cost_bps_oneway`` one-way on an assumed
    ~10 %/yr two-sided rebalance turnover (≈ 0.83 %/mo each side) — deliberately conservative for
    funds you mostly buy-and-hold. The embedded leverage cost is already inside the total return,
    so no extra borrow is levied. Returns gross vs net excess-of-cash mean, HAC t, and Sharpe.
    """
    basket = equal_weight_basket(basket_members_panel, members)
    df = pd.concat(
        [basket.rename("b"), spy_ret.rename("s"), cash_ret.rename("c")], axis=1, sort=True
    ).dropna()
    gross_ex = (df["b"] - df["c"]).to_numpy()
    # flat monthly rebalance charge: two-sided turnover × one-way cost
    monthly_turnover = 0.10 / MONTHS * 2.0        # ~10%/yr two-sided
    charge_m = monthly_turnover * cost_bps_oneway / 1e4
    net_ex = gross_ex - charge_m
    g_mean, g_t = nw_mean_t(gross_ex, lags)
    n_mean, n_t = nw_mean_t(net_ex, lags)
    return {
        "n": int(len(df)),
        "gross_exret_bps": g_mean * 1e4, "t_gross": g_t, "gross_sharpe": ann_sharpe(gross_ex),
        "net_exret_bps": n_mean * 1e4, "t_net": n_t, "net_sharpe": ann_sharpe(net_ex),
        "charge_bps_per_mo": charge_m * 1e4,
        "spy_sharpe": ann_sharpe((df["s"] - df["c"]).to_numpy()),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(world: pd.DataFrame, lags: int = NW_LAGS) -> dict:
    """Run the excess-vs-excess CAPM on a synthetic world; recover the planted net carry.

    ``alpha_ann_pct`` should be ~0 on the null (carry=roc_leak) and ~ (carry-roc_leak) when a net
    pickup is planted; ``beta`` should recover the planted leverage.
    """
    f_ex = (world["cef"] - world["cash"]).to_numpy()
    m_ex = (world["mkt"] - world["cash"]).to_numpy()
    reg = hac_ols(f_ex, m_ex, lags)
    mean_ex, t_ex = nw_mean_t(f_ex, lags)
    return {
        "alpha_ann_pct": reg["alpha"] * MONTHS * 100, "t_alpha": reg["t_alpha"],
        "beta": reg["beta"], "t_beta": reg["t_beta"], "r2": reg["r2"],
        "exret_sharpe": ann_sharpe(f_ex), "n": reg["n"],
    }
