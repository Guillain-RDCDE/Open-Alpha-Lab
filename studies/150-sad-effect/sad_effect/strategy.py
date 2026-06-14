"""The SAD seasonal strategy and its honest controls — Study 150 (SAD-Effect).

The claim (Kamstra, Kramer & Levi 2003): equity returns follow the seasonal cycle of
daylight. Autumn months (Sep–Nov, days shortening) see *depressed* returns as seasonal
risk aversion rises; the recovery phase (Dec–Mar, returning daylight) sees *elevated*
returns. The paper explains the Halloween / Sell-in-May effect and January seasonality
as manifestations of the SAD mechanism.

Three tests, one honest baseline each:

1. **Seasonal split (onset vs recovery vs unconditional).**  Split monthly returns into
   SAD-onset months (Sep–Nov), SAD-recovery months (Dec–Mar), and summer months (Apr–Aug).
   Compare each group mean to the unconditional monthly mean via a Welch t-test. The
   honest baseline is the *unconditional* mean — not "vs zero" — because equities drift
   up over time and any positive return in SAD-onset is not an edge, it's just the equity
   risk premium. The prediction: onset < unconditional < recovery.

2. **Daylight-regression (KKL's core test).**  Regress monthly returns on the
   month-over-month change in day-length hours (``delta_daylight_h``). The prediction:
   the slope coefficient should be *positive* (more daylight → higher returns). We
   report the OLS slope, its HAC (Newey-West) t-stat, and R². The honest control is the
   *null regression* (intercept only): we ask whether daylight adds explanatory power
   beyond the unconditional mean, via the t-stat on the slope coefficient.

3. **SAD calendar strategy vs buy-and-hold.**  Hold equities only in recovery months
   (Dec–Mar), cash otherwise. Compare its annualised Sharpe to buy-and-hold. The
   honest baseline is buy-and-hold, not zero: even if the strategy has a positive
   Sharpe, buy-and-hold may dominate it once opportunity cost is accounted for.

All inference uses HAC (Newey-West) t-statistics via an in-module implementation so
the engine has no hard dependency on ``quantlab`` being importable in tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import SAD_ONSET, SAD_RECOVERY

# Summer months (neither onset nor recovery in KKL's framework)
SUMMER = {4, 5, 6, 7, 8}


# ---------------------------------------------------------------------------
# Month-group labels
# ---------------------------------------------------------------------------
def month_group(index: pd.DatetimeIndex) -> pd.Series:
    """Label each date as 'onset' (Sep-Nov), 'recovery' (Dec-Mar), or 'summer' (Apr-Aug).

    This is the KKL seasonal partition. Tests on the sign and magnitude of
    onset-vs-summer and recovery-vs-summer differences are the primary verdict tests.
    """
    months = pd.DatetimeIndex(index).month
    labels = []
    for m in months:
        if m in SAD_ONSET:
            labels.append("onset")
        elif m in SAD_RECOVERY:
            labels.append("recovery")
        else:
            labels.append("summer")
    return pd.Series(labels, index=index, name="group")


# ---------------------------------------------------------------------------
# Test 1: Seasonal split
# ---------------------------------------------------------------------------
def seasonal_split(ret: pd.Series) -> dict:
    """Mean monthly return and a Welch t-test for each SAD group vs the unconditional mean.

    Returns a dict with mean, n, and Welch t (vs unconditional) for each of
    onset / recovery / summer, plus the overall unconditional mean. The prediction
    from KKL: onset_mean < unconditional < recovery_mean.
    """
    r = pd.Series(ret).astype(float).dropna()
    g = month_group(r.index)
    mu_all = float(r.mean())
    var_all = float(r.var(ddof=1))
    n_all = len(r)

    out: dict = {"unconditional_bp": mu_all * 1e4, "n_all": n_all}
    for label in ("onset", "recovery", "summer"):
        mask = (g == label).values
        sub = r.values[mask]
        n = int(mask.sum())
        mu = float(sub.mean()) if n > 0 else float("nan")
        var = float(sub.var(ddof=1)) if n > 1 else float("nan")
        # Welch t vs unconditional (all months)
        se2 = var / n + var_all / n_all
        se = np.sqrt(max(se2, 0.0))
        t = float((mu - mu_all) / se) if se > 0 and np.isfinite(mu) else float("nan")
        out[f"{label}_bp"] = mu * 1e4 if np.isfinite(mu) else float("nan")
        out[f"{label}_n"] = n
        out[f"{label}_t_vs_unconditional"] = t
    return out


# ---------------------------------------------------------------------------
# Test 2: Daylight regression (KKL core test)
# ---------------------------------------------------------------------------
def daylight_regression(ret: pd.Series, delta_dl: pd.Series) -> dict:
    """OLS regression of monthly returns on change-in-daylight-hours.

    ``delta_dl`` must be aligned on the same index as ``ret`` (use
    ``data.daylight_proxy``). Returns OLS slope, intercept, R², and the HAC
    Newey-West t-stat for the slope.

    The KKL prediction: slope > 0 (lengthening days → rising returns; shortening
    days → falling returns). A positive slope with |HAC t| >= 2 would be
    consistent with the SAD mechanism; anything below that is noise.
    """
    df = pd.DataFrame({"r": ret, "x": delta_dl}).dropna()
    y = df["r"].to_numpy(dtype=float)
    x = df["x"].to_numpy(dtype=float)
    n = len(y)
    if n < 10:
        return {k: float("nan") for k in ("slope", "intercept", "r_squared", "slope_tstat_hac", "n")}

    # OLS
    xbar, ybar = x.mean(), y.mean()
    slope = float(((x - xbar) @ (y - ybar)) / ((x - xbar) @ (x - xbar)))
    intercept = float(ybar - slope * xbar)
    resid = y - (slope * x + intercept)
    ss_res = float(resid @ resid)
    ss_tot = float(((y - ybar) ** 2).sum())
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # HAC (Newey-West) standard error for the slope
    # Asymptotic var(slope) = S_{xx}^{-2} * LRV(x * resid)
    xe = (x - xbar) * resid
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(xe @ xe) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(xe[k:] @ xe[:-k]) / n
    sxx = float((x - xbar) @ (x - xbar))
    var_slope = max(lrv, 0.0) * n / (sxx ** 2)
    se_slope = np.sqrt(var_slope)
    t_slope = float(slope / se_slope) if se_slope > 0 else float("nan")

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r2,
        "slope_tstat_hac": t_slope,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Test 3: SAD calendar strategy vs buy-and-hold
# ---------------------------------------------------------------------------
def sad_strategy(ret: pd.Series) -> pd.Series:
    """Hold equities only in SAD-recovery months (Dec–Mar), cash otherwise.

    Returns a monthly return Series aligned on ``ret``'s index. Non-recovery months
    return 0 (cash). No look-ahead: the month label is determined by the calendar
    date of the return observation — there is no forward reference.
    """
    r = pd.Series(ret).astype(float).dropna()
    months = pd.DatetimeIndex(r.index).month
    mask = np.isin(months, list(SAD_RECOVERY)).astype(float)
    return (mask * r).rename("sad_strategy")


def buy_hold(ret: pd.Series) -> pd.Series:
    """Unconditional buy-and-hold (the honest baseline for the calendar strategy)."""
    return pd.Series(ret).astype(float).dropna().rename("buy_hold")


def summary(returns: pd.Series, periods_per_year: int = 12) -> dict:
    """Annualised Sharpe, CAGR, vol, max-drawdown, and HAC t-stat for a monthly series.

    The HAC t-stat tests whether the mean return is distinguishable from zero; for a
    strategy vs buy-and-hold comparison the relevant statistic is the Sharpe ratio and
    CAGR, not the mean level (since buy-and-hold is also non-zero in expectation).
    """
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    if n < 2:
        return {k: float("nan") for k in ("sharpe", "cagr", "vol_ann", "max_dd", "n", "tstat")}
    mean, std = float(r.mean()), float(r.std(ddof=1))
    eq = (1.0 + r).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    years = n / periods_per_year
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0) if eq.iloc[-1] > 0 else float("nan")
    sharpe = float(mean / std * np.sqrt(periods_per_year)) if std > 0 else float("nan")

    # HAC t-stat on the mean
    e = r.to_numpy(dtype=float) - mean
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    tstat = float(mean / se) if se > 0 else float("nan")

    return {
        "sharpe": sharpe,
        "cagr": cagr,
        "vol_ann": float(std * np.sqrt(periods_per_year)),
        "max_dd": dd,
        "n": n,
        "tstat": tstat,
    }


# ---------------------------------------------------------------------------
# Shuffled-label control (for the synthetic positive-control demo)
# ---------------------------------------------------------------------------
def shuffled_group_returns(ret: pd.Series, seed: int = 0) -> pd.Series:
    """Shuffle the month labels while keeping the return sequence intact.

    This is the null control for the seasonal split: with shuffled labels, the
    onset/recovery groups have the same expected mean as the unconditional, so any
    apparent seasonal gap on a null synthetic tape should be small.
    """
    rng = np.random.default_rng(seed)
    idx_shuffled = pd.DatetimeIndex(rng.permutation(ret.index.to_numpy()))
    return pd.Series(ret.values, index=idx_shuffled, name="ret_shuffled").sort_index()
