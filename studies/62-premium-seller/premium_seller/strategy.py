"""Covered-call vs the underlying — the total-return spread and the upside/downside capture.

A covered-call fund earns the underlying's return minus the calls it sold (capped upside) plus the
premium. The tests: does it beat the underlying on total return and Sharpe, and what's its capture —
how much of the up months it keeps vs how much of the down months it avoids?
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


def capture(returns: pd.DataFrame, fund: str, underlying: str) -> dict:
    """Mean fund return in up vs down months of the underlying (upside/downside capture ratios)."""
    df = returns[[fund, underlying]].dropna()
    up = df[underlying] > 0
    return {"up_fund": float(df[fund][up].mean()), "up_underlying": float(df[underlying][up].mean()),
            "down_fund": float(df[fund][~up].mean()), "down_underlying": float(df[underlying][~up].mean()),
            "upside_capture": float(df[fund][up].mean() / df[underlying][up].mean()),
            "downside_capture": float(df[fund][~up].mean() / df[underlying][~up].mean())}


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
