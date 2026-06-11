"""High-dividend vs the market — the spread, the legs, and the honest total-return comparison.

The test: does a high-dividend tilt (VYM) beat the plain market (SPY) on a **total-return** basis — both
the spread (VYM − SPY) and each leg's Sharpe? If high-yield merely matches or trails total return, the
"income premium" is an illusion (you can always make your own dividend by selling shares).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12


def spread(returns: pd.DataFrame, a: str, b: str) -> pd.Series:
    """Monthly ``a − b`` total return (e.g. high-dividend minus market), aligned."""
    df = returns[[a, b]].dropna()
    return (df[a] - df[b]).rename("spread")


def spread_stats(s: pd.Series, periods_per_year: int = MONTHS) -> dict:
    """Annualised mean, Sharpe, Lo (2002) t-stat, hit-rate of a monthly spread series."""
    r = pd.Series(s).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("mean_ann", "sharpe", "tstat", "hit_rate", "n")}
    sr_m = r.mean() / r.std(ddof=1) if r.std(ddof=1) > 0 else np.nan
    se = np.sqrt((1.0 + 0.5 * sr_m**2) / len(r))
    return {
        "mean_ann": float(r.mean() * periods_per_year),
        "sharpe": float(sr_m * np.sqrt(periods_per_year)),
        "tstat": float(sr_m / se) if se > 0 else np.nan,
        "hit_rate": float((r > 0).mean()),
        "n": int(len(r)),
    }


def leg_summary(returns: pd.DataFrame, col: str) -> dict:
    return summary(returns[col].dropna())


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
