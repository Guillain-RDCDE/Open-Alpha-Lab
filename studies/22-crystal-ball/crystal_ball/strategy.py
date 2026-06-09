"""The investable book — trade the HP cycle's mean reversion — built two ways, honest and not.

The §8.1 idea: detrend with an HP filter, then trade the detrended series. The natural trade is
mean reversion of the cycle — when (log) price is *below* its HP trend (cycle < 0) it's "cheap", so go
long; when above, go short. The position is ``w_t = -sign(cycle_{t-1})``, earning today's return.

The *only* difference between the two versions of this book is which HP cycle feeds it:

    * ``causal=False`` — the two-sided cycle, which knows the future. This is the backtest as it is
      almost always (accidentally) run.
    * ``causal=True`` — the one-sided cycle, which doesn't. This is the only version you could trade.

Everything else — the rule, the costs, the benchmark — is identical, so the gap between the two books'
performance is a clean measurement of the look-ahead bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .hp import cycle

TRADING_DAYS_PER_YEAR = 252


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> dict:
    r = pd.Series(returns).astype(float).dropna()
    mean, std = r.mean(), r.std(ddof=1)
    equity = (1.0 + r).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    years = len(r) / periods_per_year
    cagr = equity.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 and equity.iloc[-1] > 0 else np.nan
    return {
        "sharpe": float(mean / std * np.sqrt(periods_per_year)) if std > 0 else np.nan,
        "ann_return": float(mean * periods_per_year),
        "vol_ann": float(std * np.sqrt(periods_per_year)),
        "cagr": float(cagr),
        "max_drawdown": float(drawdown.min()),
        "n_days": int(len(r)),
    }


def reversion_weights(close: pd.Series, lam: float = 129600.0, causal: bool = False,
                      window: int = 252) -> pd.Series:
    """Mean-reversion position ``w_t = -sign(cycle_{t-1})`` — long when price is below its HP trend.

    The cycle is lagged one bar, so the weight earns the next return. With ``causal=False`` that cycle
    is the two-sided HP (look-ahead); with ``causal=True`` it is the one-sided HP (tradable).
    """
    c = cycle(close, lam=lam, causal=causal, window=window)
    return (-np.sign(c)).shift(1).rename("weight")


def reversion_returns(close: pd.Series, cost_bps: float = 1.0, lam: float = 129600.0,
                      causal: bool = False, window: int = 252) -> pd.Series:
    """The cycle-mean-reversion stream, net of ``cost_bps`` on each change of position."""
    w = reversion_weights(close, lam=lam, causal=causal, window=window)
    r = close.pct_change()
    gross = w * r
    cost = (cost_bps * 1e-4) * w.diff().abs()
    return (gross - cost).dropna().rename("reversion")


def compare(close: pd.Series, cost_bps: float = 1.0, lam: float = 129600.0, window: int = 252,
            periods_per_year: int = TRADING_DAYS_PER_YEAR) -> dict:
    """Two-sided (look-ahead) vs one-sided (causal) cycle-reversion — the trap and the honest version.

    Aligns both books to the one-sided book's (shorter, warm-up-trimmed) sample so the comparison is
    on identical days. The gap in Sharpe is the look-ahead bias.
    """
    two = reversion_returns(close, cost_bps=cost_bps, lam=lam, causal=False)
    one = reversion_returns(close, cost_bps=cost_bps, lam=lam, causal=True, window=window)
    idx = one.index.intersection(two.index)
    s_two, s_one = summary(two.reindex(idx), periods_per_year), summary(one.reindex(idx), periods_per_year)
    return {
        "two_sided": s_two,
        "one_sided": s_one,
        "lookahead_sharpe_gap": float(s_two["sharpe"] - s_one["sharpe"]),
        "n_days": int(len(idx)),
    }
