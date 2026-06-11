"""Oil → equities: the predictive regression, the timing rule, and the controls.

Driesprong et al. (2008): last month's oil return forecasts this month's equity return, with a
**negative** sign (oil rises, stocks fall next month — a delayed reaction). The tests: (1) is the
predictive slope negative and significant, and (2) does a timing rule (hold equities after oil *falls*)
beat buy-and-hold? A simple OLS with an analytic t-stat — no scipy needed.

Two conventions, stated up front:

  * **The lag is calendar, not positional.** :func:`lag_one_month` re-dates month *m−1*'s value to
    month *m* on the monthly grid, so a hole in the index produces a NaN (dropped) instead of silently
    reaching back two or three months — the bug the old positional ``shift(1)`` had on Yahoo's gapped
    monthly CL=F feed.
  * **The cash leg earns the T-bill.** :func:`oil_timing` credits a monthly cash return when the rule
    is out of the market, and :func:`summary` takes the same ``rf`` so the timing-vs-buy-and-hold race
    is run on *excess-of-cash* Sharpe, like-for-like.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12


def lag_one_month(s: pd.Series) -> pd.Series:
    """The value dated calendar month *m−1*, re-dated to month *m* — a **calendar** lag.

    Positional ``shift(1)`` is only a one-month lag when the index has no holes; this maps values
    through the monthly period grid instead, so any missing month yields NaN rather than a stale
    two-or-three-month-old observation.
    """
    s = pd.Series(s).astype(float)
    months = pd.PeriodIndex(s.index, freq="M")
    by_month = pd.Series(s.to_numpy(), index=months + 1)  # value of m, re-dated m+1
    return pd.Series(by_month.reindex(months).to_numpy(), index=s.index)


def predict_regression(oil: pd.Series, eq: pd.Series) -> dict:
    """OLS of this month's equity return on **last calendar month's** oil return.

    Returns slope, correlation, the analytic t-stat ``r·√(n−2)/√(1−r²)`` and n. Driesprong predicts a
    *negative* slope; |t| < ~2 means no detectable relationship.
    """
    x = lag_one_month(oil)
    y = pd.Series(eq).astype(float)
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    n = len(df)
    if n < 3:
        return {"slope": np.nan, "r": np.nan, "tstat": np.nan, "n": n}
    xv, yv = df["x"].to_numpy(), df["y"].to_numpy()
    r = float(np.corrcoef(xv, yv)[0, 1])
    slope = r * yv.std(ddof=1) / xv.std(ddof=1)
    tstat = r * np.sqrt(n - 2) / np.sqrt(max(1.0 - r**2, 1e-12))
    return {"slope": float(slope), "r": r, "tstat": float(tstat), "n": n}


def oil_timing(oil: pd.Series, eq: pd.Series, tbill: pd.Series | None = None) -> pd.Series:
    """Hold equities this month iff last calendar month's oil fell, else hold cash.

    Cash is **credited**, not zeroed: pass ``tbill`` (a monthly cash-return series, e.g. the ``tbill``
    column of :func:`black_gold.data.fetch_pair`) and the out-of-market months earn it. ``tbill=None``
    leaves cash at 0 (the synthetic world has no cash rate). Months with no lagged signal are dropped.
    """
    eqs = pd.Series(eq).astype(float)
    lag = lag_one_month(oil).reindex(eqs.index)
    if tbill is None:
        cash = pd.Series(0.0, index=eqs.index)
    else:
        cash = pd.Series(tbill).astype(float).reindex(eqs.index).fillna(0.0)
    return eqs.where(lag < 0, cash)[lag.notna()].rename("oil_timing")


def time_in_market(oil: pd.Series) -> float:
    lag = lag_one_month(oil)
    return float((lag.dropna() < 0).mean())


def buy_hold(eq: pd.Series) -> pd.Series:
    return pd.Series(eq).astype(float).dropna().rename("buy_hold")


def summary(returns: pd.Series, periods_per_year: int = MONTHS, rf: pd.Series | None = None) -> dict:
    """Annualised Sharpe, CAGR, vol, max-drawdown for a monthly return series.

    **Sharpe convention**: raw (``mean/std``) when ``rf`` is None; with ``rf`` (a monthly cash-return
    series) it is the standard **excess-of-cash** Sharpe, ``mean(r−rf)/std(r−rf)``. Pass the *same*
    ``rf`` to both legs of a race (timing vs buy-and-hold) so the comparison is like-for-like.
    CAGR / vol / max-drawdown always describe the raw series.
    """
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("sharpe", "cagr", "vol_ann", "max_drawdown", "n")}
    if rf is None:
        ex = r
    else:
        ex = (r - pd.Series(rf).astype(float).reindex(r.index).fillna(0.0)).dropna()
    ex_mean, ex_std = ex.mean(), ex.std(ddof=1)
    std = r.std(ddof=1)
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax() - 1.0).min()
    years = len(r) / periods_per_year
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0 if eq.iloc[-1] > 0 else np.nan
    return {
        "sharpe": float(ex_mean / ex_std * np.sqrt(periods_per_year)) if ex_std > 0 else np.nan,
        "cagr": float(cagr),
        "vol_ann": float(std * np.sqrt(periods_per_year)),
        "max_drawdown": float(dd),
        "n": int(len(r)),
    }
