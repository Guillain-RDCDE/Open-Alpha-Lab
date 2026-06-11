"""Oil → equities: the predictive regression, the timing rule, and the controls.

Driesprong et al. (2008): last month's oil return forecasts this month's equity return, with a
**negative** sign (oil rises, stocks fall next month — a delayed reaction). The tests: (1) is the
predictive slope negative and significant, and (2) does a timing rule (hold equities after oil *falls*)
beat buy-and-hold? A simple OLS with an analytic t-stat — no scipy needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12


def predict_regression(oil: pd.Series, eq: pd.Series) -> dict:
    """OLS of this month's equity return on **last month's** oil return.

    Returns slope, correlation, the analytic t-stat ``r·√(n−2)/√(1−r²)`` and n. Driesprong predicts a
    *negative* slope; |t| < ~2 means no detectable relationship.
    """
    x = pd.Series(oil).astype(float).shift(1)
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


def oil_timing(oil: pd.Series, eq: pd.Series) -> pd.Series:
    """Hold equities this month iff last month's oil return was negative (oil fell), else cash."""
    pos = (pd.Series(oil).astype(float).shift(1) < 0).astype(float)
    return (pos * pd.Series(eq).astype(float)).dropna().rename("oil_timing")


def time_in_market(oil: pd.Series) -> float:
    return float((pd.Series(oil).astype(float).shift(1) < 0).mean())


def buy_hold(eq: pd.Series) -> pd.Series:
    return pd.Series(eq).astype(float).dropna().rename("buy_hold")


def summary(returns: pd.Series, periods_per_year: int = MONTHS) -> dict:
    """Annualised Sharpe, CAGR, vol, max-drawdown for a monthly return series."""
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("sharpe", "cagr", "vol_ann", "max_drawdown", "n")}
    mean, std = r.mean(), r.std(ddof=1)
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax() - 1.0).min()
    years = len(r) / periods_per_year
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0 if eq.iloc[-1] > 0 else np.nan
    return {
        "sharpe": float(mean / std * np.sqrt(periods_per_year)) if std > 0 else np.nan,
        "cagr": float(cagr),
        "vol_ann": float(std * np.sqrt(periods_per_year)),
        "max_drawdown": float(dd),
        "n": int(len(r)),
    }
