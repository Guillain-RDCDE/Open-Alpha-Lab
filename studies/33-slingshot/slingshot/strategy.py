"""The book — short-horizon cross-sectional mean-reversion, dollar-neutral within one group.

Each day:
  1. **Signal** — each stock's trailing ``lookback``-day return, demeaned across the cross-section
     (its return *relative to the pack*), then negated: short the relative leaders, long the relative
     laggards. The book runs on a **two-day execution lag** (the signal is shifted once at
     construction in :func:`reversal_signal` and once more at execution in :func:`book_returns`) —
     conservative: every headline number is earned trading a day *later* than the fastest tradable
     implementation.
  2. **Dollar-neutral weights** — scale the demeaned signal so the long and short legs net to zero and
     gross exposure sums to one. No single-name or sector cap (a fork in beat 7).
  3. **Daily rebalance** — the signal refreshes every day, so the book turns over almost completely;
     that turnover is the whole tradability question.

The deliberate contrast with Study 32 (Rip-Tide): there we faded *futures* against their own past and
found no premium at all. Here we fade *stocks* against each other — where short-term reversal is a real,
documented anomaly — and ask whether the gross edge can clear its own transaction costs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def reversal_signal(returns: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Dollar-neutral contrarian weights: the cross-sectionally demeaned, negated trailing return,
    normalised so gross exposure is 1 each day, shifted one day here (and once more at execution in
    :func:`book_returns` — a two-day lag in total, conservative). Long relative laggards, short
    relative leaders."""
    prices = (1.0 + returns.fillna(0.0)).cumprod()
    trail = prices / prices.shift(lookback) - 1.0
    x = trail.sub(trail.mean(axis=1), axis=0)              # relative to the cross-section
    w = -x                                                  # fade the relative move
    gross = w.abs().sum(axis=1).replace(0.0, np.nan)
    w = w.div(gross, axis=0).fillna(0.0)
    return w.shift(1).fillna(0.0)


def book_returns(returns: pd.DataFrame, lookback: int = 5, cost_bps: float = 5.0) -> pd.Series:
    """Net daily return of the dollar-neutral reversal book, minus turnover cost (``cost_bps`` per unit
    traded). The weights from :func:`reversal_signal` (already shifted once) are shifted once more here —
    a **two-day execution lag** in total, which is conservative (a faster fill could only help a reversal
    signal). Equities cost more to trade than futures (~5 bp here), and the book rebalances daily, so the
    cost term is decisive."""
    w = reversal_signal(returns, lookback=lookback)
    gross = (w.shift(1) * returns).sum(axis=1)
    cost = (cost_bps * 1e-4) * w.diff().abs().sum(axis=1)
    return (gross - cost).rename("reversion")


def turnover(returns: pd.DataFrame, lookback: int = 5) -> float:
    """Average daily one-way turnover (Σ|Δw|) — for a daily-rebalanced dollar-neutral book this is large,
    which is exactly why the net edge is in doubt."""
    w = reversal_signal(returns, lookback=lookback)
    return float(w.diff().abs().sum(axis=1).mean())


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Annualised Sharpe, CAGR, vol, max-drawdown, Calmar, skew for a daily return series."""
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("sharpe", "cagr", "vol_ann", "max_drawdown", "calmar", "skew", "n_days")}
    mean, std = r.mean(), r.std(ddof=1)
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax() - 1.0).min()
    years = len(r) / periods_per_year
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0 if eq.iloc[-1] > 0 else np.nan
    return {"sharpe": float(mean / std * np.sqrt(periods_per_year)) if std > 0 else np.nan,
            "cagr": float(cagr), "vol_ann": float(std * np.sqrt(periods_per_year)),
            "max_drawdown": float(dd), "calmar": float(cagr / abs(dd)) if dd < 0 else np.nan,
            "skew": float(r.skew()), "n_days": int(len(r))}
