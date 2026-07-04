"""Strategy & stats for Study 610 — Fallen-Angels-Premium.

The claim (Ben Dor & Xu 2011; Arbess & Rosa index studies since 1997; the ICE/BofA
fallen-angel index literature): bonds ejected from investment-grade indices are dumped
by forced sellers (IG mandates *must* sell on downgrade), get oversold, and then
outperform the broad high-yield market. The live test: **ANGL vs HYG** (2012→) with
**FALN vs HYG** (2016→) as the independent confirmation.

The machinery, all offline once the cache exists:

* ``monthly_returns`` — month-end total returns, final partial month dropped.
* ``nw_ols`` — OLS with Newey-West (Bartlett/HAC) standard errors; the workhorse for
  the excess-spread HAC t (intercept-only), the CAPM-style alpha (one factor) and the
  duration/quality honesty check (HY + rate factors).
* ``capm_alpha`` / ``factor_alpha`` — excess-vs-excess regressions on the BIL
  (tradable T-bill) risk-free proxy.
* ``sharpe_excess`` — annualised excess-return Sharpe (excess-vs-excess race only).
* ``max_drawdown`` — daily total-return drawdown depth + dates.
* ``switch_cost_drag`` — the one-off HYG→ANGL round-trip switch amortised per year.

No RNG anywhere in the real-tape path — every number is a deterministic function of
the cached tape. The only randomness in the study lives in the synthetic world
(``data.synthetic_world``), which is fixed-seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Returns
# --------------------------------------------------------------------------- #
def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Wide monthly simple total returns (month-end to month-end).

    Drops the final calendar month if the price tape ends before that month is over,
    so a stamped run never contains a partial month.
    """
    m = prices.resample("ME").last()
    ret = m.pct_change()
    last_px = prices.index.max()
    last_bucket = ret.index.max()
    if last_px < (last_bucket - pd.offsets.MonthEnd(0)) or last_px.day < last_bucket.day:
        ret = ret.iloc[:-1]
    return ret


# --------------------------------------------------------------------------- #
# Newey-West OLS (the single inference engine)
# --------------------------------------------------------------------------- #
def nw_ols(y: pd.Series, X: pd.DataFrame | None = None,
           lags: int | None = None) -> dict:
    """OLS of ``y`` on a constant (+ optional factor columns) with HAC errors.

    Newey-West long-run covariance with the Bartlett kernel; ``lags=None`` uses the
    rule of thumb ``floor(4*(n/100)^(2/9))``. With ``X=None`` this is the HAC t of
    the mean (the monthly excess-spread test). Returns coefficient estimates in
    bps/month for the intercept and unitless betas for the factors, each with its
    HAC t-stat.
    """
    if X is not None:
        df = pd.concat([y.rename("_y"), X], axis=1).dropna()
        yv = df["_y"].to_numpy(dtype=float)
        names = list(X.columns)
        Xm = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(dtype=float) for c in names])
    else:
        yv = pd.Series(y).dropna().to_numpy(dtype=float)
        names = []
        Xm = np.ones((yv.size, 1))
    n, k = Xm.shape
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    beta = np.linalg.solve(Xm.T @ Xm, Xm.T @ yv)
    e = yv - Xm @ beta
    # HAC "meat": S = 1/n sum_j w_j sum_t (x_t e_t)(x_{t-j} e_{t-j})' (+ transpose)
    Z = Xm * e[:, None]
    S = (Z.T @ Z) / n
    for j in range(1, lags + 1):
        w = 1.0 - j / (lags + 1.0)
        G = (Z[j:].T @ Z[:-j]) / n
        S += w * (G + G.T)
    XtX_inv = np.linalg.inv(Xm.T @ Xm / n)
    cov = XtX_inv @ S @ XtX_inv / n
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    t = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    out = {
        "alpha_bps": beta[0] * 1e4,
        "alpha_se_bps": se[0] * 1e4,
        "t_alpha": t[0],
        "n": n,
        "lags": lags,
        "r2": 1.0 - (e @ e) / max(((yv - yv.mean()) @ (yv - yv.mean())), 1e-30),
    }
    for i, name in enumerate(names, start=1):
        out[f"beta_{name}"] = beta[i]
        out[f"t_{name}"] = t[i]
    return out


def hac_mean(series: pd.Series, lags: int | None = None) -> dict:
    """HAC (Newey-West) t of the mean of a monthly series — the excess-spread test."""
    return nw_ols(series, X=None, lags=lags)


# --------------------------------------------------------------------------- #
# Alpha regressions (excess-vs-excess on the tradable BIL risk-free)
# --------------------------------------------------------------------------- #
def excess(monthly: pd.DataFrame, col: str, rf: str = "BIL") -> pd.Series:
    """Monthly excess return of ``col`` over the T-bill ETF proxy."""
    return (monthly[col] - monthly[rf]).dropna()


def capm_alpha(monthly: pd.DataFrame, asset: str, bench: str,
               rf: str = "BIL") -> dict:
    """CAPM-style alpha of ``asset`` over ``bench`` (both in excess of ``rf``)."""
    y = excess(monthly, asset, rf)
    x = excess(monthly, bench, rf).rename(bench)
    return nw_ols(y, x.to_frame())


def factor_alpha(monthly: pd.DataFrame, asset: str, factors: list[str],
                 rf: str = "BIL") -> dict:
    """Multi-factor alpha: ``asset`` excess on several factor excess returns.

    The duration/quality honesty check regresses ANGL excess on HYG excess (credit
    beta) + IEF excess (rate duration); if the one-factor alpha is just a duration
    tilt in disguise, it dies here.
    """
    y = excess(monthly, asset, rf)
    X = pd.concat([excess(monthly, f, rf).rename(f) for f in factors], axis=1)
    return nw_ols(y, X)


# --------------------------------------------------------------------------- #
# Performance & risk
# --------------------------------------------------------------------------- #
def ann_return(monthly_ret: pd.Series) -> float:
    """Geometric annualised return (%) from monthly simple returns."""
    r = monthly_ret.dropna()
    return ((1.0 + r).prod() ** (12.0 / len(r)) - 1.0) * 100.0


def sharpe_excess(monthly: pd.DataFrame, asset: str, rf: str = "BIL") -> float:
    """Annualised Sharpe of monthly EXCESS returns (excess-vs-excess race only)."""
    e = excess(monthly, asset, rf)
    return float(e.mean() / e.std(ddof=1) * np.sqrt(12.0))


def max_drawdown(prices: pd.Series) -> dict:
    """Max drawdown of a daily total-return price series: depth (%), peak/trough dates."""
    px = prices.dropna()
    peak = px.cummax()
    dd = px / peak - 1.0
    trough = dd.idxmin()
    peak_date = px.loc[:trough].idxmax()
    return {"depth_pct": float(dd.min() * 100.0),
            "peak": str(peak_date.date()), "trough": str(trough.date())}


def switch_cost_drag(spread_bps_oneway: float, years: float) -> float:
    """Annualised drag (bps/yr) of ONE round-trip switch (sell HYG, buy ANGL).

    Costs are one-way x NAV on each leg: 2 legs x ``spread_bps_oneway`` amortised
    over the holding period. Long-only, no shorts, no borrow.
    """
    return 2.0 * spread_bps_oneway / years


def welch_t(a: pd.Series, b: pd.Series) -> dict:
    """Welch t of the difference in means between two independent samples.

    Used on the half-split of the monthly ANGL-HYG spread: a decay claim needs a
    significant *difference* between halves, not just one half dipping under t=2.
    """
    av, bv = pd.Series(a).dropna(), pd.Series(b).dropna()
    na, nb = len(av), len(bv)
    va, vb = av.var(ddof=1), bv.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    return {"diff_bps": (av.mean() - bv.mean()) * 1e4,
            "t": float((av.mean() - bv.mean()) / se), "n_a": na, "n_b": nb}


def align_common(monthly: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Rows where every column in ``cols`` has a return (the common sample)."""
    return monthly[cols].dropna()
