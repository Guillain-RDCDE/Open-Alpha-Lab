"""Strategy and analysis layer for Study 276 (Sneaker-Resale).

The claim under test: *the sneaker resale market (StockX, GOAT) is a real alternative
asset class that "beats stocks."* We pin the curated annual sneaker-resale index
against ^GSPC (S&P 500 price index) calendar-year returns and ask three honest
questions:

  1. **Does it beat stocks on return?** Test the mean annual excess return
     (sneaker - S&P) against zero with a Welch / one-sample t-test AND a HAC
     (Newey-West) t-stat — the REAL-tape bar for a Signal verdict is |HAC t| >= 2.

  2. **Is its high Sharpe real, or a stale-pricing artifact?** The index is an
     appraisal-style annual series. We measure its first-order autocorrelation and
     apply a Geltner (1991, 1993) AR(1) **unsmoothing** to recover the economic
     volatility. If the Sharpe collapses after unsmoothing, the "low-risk
     outperformance" is an illusion.

  3. **Could you actually trade it?** Physical sneaker resale carries enormous
     one-way frictions (marketplace seller fees ~10-15%, authentication, shipping,
     storage, no shorting, no fund), and there is no investable index vehicle. We
     net these out and label gross vs net, price-only vs total-return, and the
     survivorship of the curated basket on the Signal axis.

n is tiny (18 annual observations, 2007-2024). With ~18-20% S&P volatility, the
minimum detectable annual excess at |t| = 2 is ~9%/yr — far above anything observed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Newey-West (HAC) t-stat for a mean — the REAL-tape significance bar
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray, lags: int = 1) -> tuple[float, float]:
    """Newey-West HAC t-stat for H0: mean(x) = 0.

    Returns ``(t, se)``. With ``lags`` Bartlett-weighted autocovariances, this is the
    honest standard error for a serially-correlated annual series (the sneaker excess
    return is autocorrelated because the index is smoothed).
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return float("nan"), float("nan")
    m = x.mean()
    e = x - m
    gamma0 = float(e @ e) / n
    lrv = gamma0
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        gk = float(e[k:] @ e[:-k]) / n
        lrv += 2.0 * w * gk
    lrv = max(lrv, 1e-12)
    se = np.sqrt(lrv / n)
    return float(m / se), float(se)


# ---------------------------------------------------------------------------
# Geltner AR(1) unsmoothing — undo the stale-appraisal autocorrelation
# ---------------------------------------------------------------------------
def unsmooth_ar1(returns: pd.Series) -> tuple[pd.Series, float]:
    """Geltner (1991) AR(1) unsmoothing of an appraisal-style return series.

    Given observed (smoothed) returns ``r_obs`` with first-order autocorrelation
    ``rho``, the recovered economic return is
    ``r_true_t = (r_obs_t - rho * r_obs_{t-1}) / (1 - rho)``.

    Returns ``(unsmoothed_series, rho)``. If ``rho <= 0`` (no positive smoothing),
    the series is returned essentially unchanged.
    """
    r = returns.dropna()
    if len(r) < 3:
        return r.copy(), float("nan")
    rho = float(r.autocorr(lag=1))
    if not np.isfinite(rho) or rho <= 0.0:
        return r.copy(), rho
    ru = (r - rho * r.shift(1)) / (1.0 - rho)
    return ru.dropna(), rho


# ---------------------------------------------------------------------------
# Core comparison: sneaker vs S&P
# ---------------------------------------------------------------------------
def compare_returns(
    df: pd.DataFrame,
    sneaker_col: str = "sneaker_return",
    gspc_col: str = "gspc_return",
    one_way_cost: float = 0.03,
    rf: float = 0.0,
) -> dict:
    """Compare the sneaker index against the S&P on return, risk, and cost.

    Parameters
    ----------
    df : DataFrame with annual ``sneaker_return`` and ``gspc_return`` columns.
    one_way_cost : annual frictional haircut applied to the sneaker leg to proxy
        marketplace fees / authentication / shipping / storage on physical resale.
    rf : annual risk-free rate for the Sharpe ratios (default 0).

    Returns a flat dict of headline numbers: means, vols, CAGRs, Sharpes (raw and
    unsmoothed), the excess-return t-stats (OLS and HAC), the AR(1) smoothing
    coefficient, and the net-of-cost sneaker mean.
    """
    sn = df[sneaker_col].to_numpy(dtype=float)
    gs = df[gspc_col].to_numpy(dtype=float)
    n = len(sn)

    sn_mean, sn_vol = float(np.mean(sn)), float(np.std(sn, ddof=1))
    gs_mean, gs_vol = float(np.mean(gs)), float(np.std(gs, ddof=1))

    sn_cagr = float(np.prod(1.0 + sn) ** (1.0 / n) - 1.0)
    gs_cagr = float(np.prod(1.0 + gs) ** (1.0 / n) - 1.0)

    excess = sn - gs
    t_ols, p_ols = scipy_stats.ttest_1samp(excess, 0.0)
    t_hac, _ = hac_tstat(excess, lags=1)

    sn_sharpe = (sn_mean - rf) / sn_vol if sn_vol > 0 else float("nan")
    gs_sharpe = (gs_mean - rf) / gs_vol if gs_vol > 0 else float("nan")

    # Unsmoothed sneaker Sharpe — the honest, stale-pricing-adjusted version.
    sn_unsm, rho = unsmooth_ar1(df[sneaker_col])
    if len(sn_unsm) >= 2:
        u_mean = float(sn_unsm.mean())
        u_vol = float(sn_unsm.std(ddof=1))
        sn_unsm_sharpe = (u_mean - rf) / u_vol if u_vol > 0 else float("nan")
    else:
        u_vol, sn_unsm_sharpe = float("nan"), float("nan")

    net_mean = sn_mean - one_way_cost

    return {
        "n": int(n),
        "sneaker_mean": round(sn_mean, 4),
        "sneaker_vol": round(sn_vol, 4),
        "sneaker_cagr": round(sn_cagr, 4),
        "sneaker_sharpe": round(sn_sharpe, 4),
        "gspc_mean": round(gs_mean, 4),
        "gspc_vol": round(gs_vol, 4),
        "gspc_cagr": round(gs_cagr, 4),
        "gspc_sharpe": round(gs_sharpe, 4),
        "excess_mean": round(float(np.mean(excess)), 4),
        "excess_t_ols": round(float(t_ols), 3),
        "excess_p_ols": round(float(p_ols), 4),
        "excess_t_hac": round(float(t_hac), 3),
        "ar1_rho": round(float(rho), 4),
        "unsmoothed_vol": round(float(u_vol), 4),
        "unsmoothed_sharpe": round(float(sn_unsm_sharpe), 4),
        "net_sneaker_mean": round(float(net_mean), 4),
        "one_way_cost": one_way_cost,
    }


# ---------------------------------------------------------------------------
# A tradable proxy: long-the-trend with a one-year execution lag
# ---------------------------------------------------------------------------
def trend_following_excess(
    df: pd.DataFrame,
    sneaker_col: str = "sneaker_return",
    gspc_col: str = "gspc_return",
    one_way_cost: float = 0.03,
) -> dict:
    """A momentum proxy: hold the sneaker basket only after an up year (lagged).

    Signal: be long the sneaker index in year t **iff** the index rose in year t-1
    (one-year execution lag — no look-ahead). Otherwise sit in the S&P. We net the
    sneaker leg of ``one_way_cost`` whenever we are switched into / out of it.

    Returns annualised strategy CAGR vs an always-S&P benchmark — to show that even a
    'smart' timing overlay does not rescue the idea.
    """
    d = df.sort_values("year").reset_index(drop=True)
    sn = d[sneaker_col].to_numpy(dtype=float)
    gs = d[gspc_col].to_numpy(dtype=float)
    n = len(sn)

    in_sneaker = np.zeros(n, dtype=bool)
    for t in range(1, n):
        in_sneaker[t] = sn[t - 1] > 0.0  # lagged signal

    strat = np.where(in_sneaker, sn - one_way_cost, gs)
    strat_cagr = float(np.prod(1.0 + strat) ** (1.0 / n) - 1.0)
    bah_cagr = float(np.prod(1.0 + gs) ** (1.0 / n) - 1.0)
    n_in = int(in_sneaker.sum())

    return {
        "strat_cagr": round(strat_cagr, 4),
        "bah_gspc_cagr": round(bah_cagr, 4),
        "n_years_in_sneaker": n_in,
        "n_total": n,
        "shortfall_pp": round((strat_cagr - bah_cagr) * 100, 2),
    }


# ---------------------------------------------------------------------------
# Summarize — compact R-dict for notebooks / results.md
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame, one_way_cost: float = 0.03) -> dict:
    """One-stop summary dict for the real merged DataFrame.

    Returns a flat dict mirroring the R = dict(...) in build_notebooks.py and
    docs/results.md. Percent fields are in percent units (e.g. 7.5 == 7.5%).
    """
    c = compare_returns(df, one_way_cost=one_way_cost)
    tf = trend_following_excess(df, one_way_cost=one_way_cost)
    return {
        "n": c["n"],
        "sneaker_cagr_pct": round(c["sneaker_cagr"] * 100, 1),
        "gspc_cagr_pct": round(c["gspc_cagr"] * 100, 1),
        "sneaker_mean_pct": round(c["sneaker_mean"] * 100, 1),
        "sneaker_vol_pct": round(c["sneaker_vol"] * 100, 1),
        "gspc_mean_pct": round(c["gspc_mean"] * 100, 1),
        "gspc_vol_pct": round(c["gspc_vol"] * 100, 1),
        "excess_mean_pct": round(c["excess_mean"] * 100, 1),
        "excess_t_ols": c["excess_t_ols"],
        "excess_p_ols": c["excess_p_ols"],
        "excess_t_hac": c["excess_t_hac"],
        "sneaker_sharpe": c["sneaker_sharpe"],
        "gspc_sharpe": c["gspc_sharpe"],
        "ar1_rho": c["ar1_rho"],
        "unsmoothed_vol_pct": round(c["unsmoothed_vol"] * 100, 1),
        "unsmoothed_sharpe": c["unsmoothed_sharpe"],
        "net_sneaker_mean_pct": round(c["net_sneaker_mean"] * 100, 1),
        "strat_cagr_pct": round(tf["strat_cagr"] * 100, 1),
        "bah_gspc_cagr_pct": round(tf["bah_gspc_cagr"] * 100, 1),
        "trend_shortfall_pp": tf["shortfall_pp"],
    }
