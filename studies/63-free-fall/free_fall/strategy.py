"""Short-vol carry vs the market — the premium, the skew, and the crash that defines it.

Shorting volatility (SVXY) earns a steady carry punctuated by rare catastrophic losses. The tests:
does it earn a premium (mean return, post-crash Sharpe), and is the left tail survivable (skew, worst
day, max drawdown)? A real carry with a ruinous tail is FRAGILE, not investable.
"""
from __future__ import annotations
import numpy as np, pandas as pd
TRADING_DAYS = 252


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Annualised Sharpe, CAGR, vol, max-drawdown, skew, worst single day for a daily return series."""
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("sharpe", "cagr", "vol_ann", "max_drawdown", "skew", "worst_day", "n")}
    mean, std = r.mean(), r.std(ddof=1)
    eq = (1.0 + r).cumprod(); dd = (eq / eq.cummax() - 1.0).min()
    cagr = eq.iloc[-1] ** (periods_per_year / len(r)) - 1.0 if eq.iloc[-1] > 0 else float("nan")
    return {"sharpe": float(mean / std * np.sqrt(periods_per_year)) if std > 0 else np.nan, "cagr": float(cagr),
            "vol_ann": float(std * np.sqrt(periods_per_year)), "max_drawdown": float(dd),
            "skew": float(r.skew()), "worst_day": float(r.min()), "n": int(len(r))}


def worst_day(returns: pd.Series) -> tuple:
    """The single worst day and its date — the steamroller moment."""
    r = pd.Series(returns).astype(float).dropna()
    return (float(r.min()), r.idxmin())


def carry_vs_crash(returns: pd.Series, crash_threshold: float = -0.20) -> dict:
    """Decompose: the median 'normal' day (the carry being collected) vs the worst crash day, and the
    share of the cumulative loss concentrated in the few crash days (|return| > ``crash_threshold``)."""
    r = pd.Series(returns).astype(float).dropna()
    crashes = r[r <= crash_threshold]
    return {"median_day_bp": float(r.median() * 1e4), "mean_day_bp": float(r.mean() * 1e4),
            "worst_day": float(r.min()), "n_crash_days": int(len(crashes)),
            "crash_days_total": float((1.0 + crashes).prod() - 1.0) if len(crashes) else 0.0}
