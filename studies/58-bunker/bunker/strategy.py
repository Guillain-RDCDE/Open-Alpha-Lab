"""Min-vol ETF vs the market — the spread, the legs, and the risk-vs-Sharpe question.

Does USMV (minimum volatility) beat SPY (market) on a risk-adjusted basis, or merely lower the risk? We
report each leg's Sharpe/vol/drawdown, the spread, and the volatility-reduction ratio.
"""
from __future__ import annotations
import numpy as np, pandas as pd
MONTHS = 12


def spread(returns: pd.DataFrame, a: str, b: str) -> pd.Series:
    df = returns[[a, b]].dropna()
    return (df[a] - df[b]).rename("spread")


def spread_stats(s: pd.Series, periods_per_year: int = MONTHS) -> dict:
    r = pd.Series(s).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("mean_ann", "sharpe", "tstat", "hit_rate", "n")}
    sr_m = r.mean() / r.std(ddof=1) if r.std(ddof=1) > 0 else np.nan
    se = np.sqrt((1.0 + 0.5 * sr_m**2) / len(r))
    return {"mean_ann": float(r.mean() * periods_per_year), "sharpe": float(sr_m * np.sqrt(periods_per_year)),
            "tstat": float(sr_m / se) if se > 0 else np.nan, "hit_rate": float((r > 0).mean()), "n": int(len(r))}


def leg_summary(returns: pd.DataFrame, col: str) -> dict:
    return summary(returns[col].dropna())


def vol_reduction(returns: pd.DataFrame, defensive: str, market: str) -> float:
    """Ratio of the defensive ETF's annualised vol to the market's (<1 ⇒ it really is calmer)."""
    a = summary(returns[defensive].dropna())["vol_ann"]
    b = summary(returns[market].dropna())["vol_ann"]
    return float(a / b) if b else np.nan


def summary(returns: pd.Series, periods_per_year: int = MONTHS) -> dict:
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("sharpe", "cagr", "vol_ann", "max_drawdown", "n")}
    mean, std = r.mean(), r.std(ddof=1)
    eq = (1.0 + r).cumprod(); dd = (eq / eq.cummax() - 1.0).min(); years = len(r) / periods_per_year
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0 if eq.iloc[-1] > 0 else np.nan
    return {"sharpe": float(mean / std * np.sqrt(periods_per_year)) if std > 0 else np.nan, "cagr": float(cagr),
            "vol_ann": float(std * np.sqrt(periods_per_year)), "max_drawdown": float(dd), "n": int(len(r))}
