"""Leveraged-ETF mechanics — daily rebalancing, volatility drag, and the regime question.

A daily L-times leveraged ETF earns ``L x today's return`` each day, compounded. Over time its CAGR
falls short of ``L x (underlying CAGR)`` by the **volatility drag** ~ ``0.5 x L x (L-1) x vol^2``. We
quantify that gap, check it against theory, and compare the leveraged product to the underlying on a
risk-adjusted basis (Sharpe, drawdown) — which is where leverage's real cost shows.
"""
from __future__ import annotations
import numpy as np, pandas as pd
TRADING_DAYS = 252


def lever_daily(returns: pd.Series, L: float = 3.0) -> pd.Series:
    """Daily L-times leveraged return series (the ETF rebalances to L x exposure each day)."""
    return (L * pd.Series(returns).astype(float)).rename(f"{L:g}x")


def cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    r = pd.Series(returns).astype(float).dropna()
    eq = (1.0 + r).prod()
    return float(eq ** (periods_per_year / len(r)) - 1.0) if eq > 0 else float("nan")


def vol_drag_theory(returns: pd.Series, L: float = 3.0, periods_per_year: int = TRADING_DAYS) -> float:
    """Theoretical annual volatility drag of an L-times daily-rebalanced ETF: 0.5*L*(L-1)*vol^2."""
    vol = pd.Series(returns).astype(float).dropna().std() * np.sqrt(periods_per_year)
    return float(0.5 * L * (L - 1.0) * vol ** 2)


def decay_gap(underlying: pd.Series, L: float = 3.0) -> dict:
    """The realized leverage decay: L*(underlying CAGR) minus the leveraged product's realized CAGR,
    next to the theoretical vol drag. ``levered`` here is the *self-replicated* L-times daily series."""
    u = pd.Series(underlying).astype(float).dropna()
    lev = lever_daily(u, L)
    naive = L * cagr(u)
    realized = cagr(lev)
    return {"underlying_cagr": cagr(u), "naive_Lx_cagr": naive, "levered_cagr": realized,
            "decay": float(naive - realized), "drag_theory": vol_drag_theory(u, L), "L": L}


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("sharpe", "cagr", "vol_ann", "max_drawdown", "n")}
    mean, std = r.mean(), r.std(ddof=1)
    eq = (1.0 + r).cumprod(); dd = (eq / eq.cummax() - 1.0).min()
    return {"sharpe": float(mean / std * np.sqrt(periods_per_year)) if std > 0 else np.nan,
            "cagr": cagr(r, periods_per_year), "vol_ann": float(std * np.sqrt(periods_per_year)),
            "max_drawdown": float(dd), "n": int(len(r))}
