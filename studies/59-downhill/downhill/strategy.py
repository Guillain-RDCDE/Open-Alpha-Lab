"""The term premium — long-duration Treasuries minus cash — and whether the risk is worth it.

Holding intermediate Treasuries (IEF, ~7-10y) instead of T-bills (BIL) earns the term premium and the
roll-down. The tests: is the excess (IEF - BIL) positive (a real premium), and is it *well paid* — i.e.
does extending duration improve the Sharpe, or just add poorly-compensated risk vs holding bills?
"""
from __future__ import annotations
import numpy as np, pandas as pd
MONTHS = 12


def term_premium(returns: pd.DataFrame, long_dur: str = "IEF", cash: str = "BIL") -> pd.Series:
    """Monthly excess return of long-duration Treasuries over cash (the term premium / roll-down)."""
    df = returns[[long_dur, cash]].dropna()
    return (df[long_dur] - df[cash]).rename("term_premium")


def excess_stats(s: pd.Series, periods_per_year: int = MONTHS) -> dict:
    r = pd.Series(s).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("mean_ann", "sharpe", "tstat", "hit_rate", "n")}
    sr_m = r.mean() / r.std(ddof=1) if r.std(ddof=1) > 0 else np.nan
    se = np.sqrt((1.0 + 0.5 * sr_m**2) / len(r))
    return {"mean_ann": float(r.mean() * periods_per_year), "sharpe": float(sr_m * np.sqrt(periods_per_year)),
            "tstat": float(sr_m / se) if se > 0 else np.nan, "hit_rate": float((r > 0).mean()), "n": int(len(r))}


def leg_summary(returns: pd.DataFrame, col: str) -> dict:
    return summary(returns[col].dropna())


def summary(returns: pd.Series, periods_per_year: int = MONTHS) -> dict:
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("sharpe", "cagr", "vol_ann", "max_drawdown", "n")}
    mean, std = r.mean(), r.std(ddof=1)
    eq = (1.0 + r).cumprod(); dd = (eq / eq.cummax() - 1.0).min(); years = len(r) / periods_per_year
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0 if eq.iloc[-1] > 0 else np.nan
    return {"sharpe": float(mean / std * np.sqrt(periods_per_year)) if std > 0 else np.nan, "cagr": float(cagr),
            "vol_ann": float(std * np.sqrt(periods_per_year)), "max_drawdown": float(dd), "n": int(len(r))}
