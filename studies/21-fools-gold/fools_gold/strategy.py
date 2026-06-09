"""The investable book — long in 'golden', in cash in 'death' — and the benchmark it has to beat.

The canonical crossover trade is **long/flat**: hold the asset while the fast MA is above the slow one,
sit in cash otherwise (Faber's timing rule). It is really a crude *trend filter*, so two things are
true at once and must be kept apart: it spends a chunk of time *out* of the market (lower average
exposure, hence lower vol and shallower drawdowns), and it pays a toll every time the averages whipsaw.
The honest question is never "does it cut drawdowns" (of course it does — it holds cash) but **does it
beat just buying and holding on a risk-adjusted, net-of-cost basis** — and whether the crossover adds
anything a constant lower exposure wouldn't.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cross import cross_state

TRADING_DAYS_PER_YEAR = 252


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> dict:
    """Headline stats for a return stream: Sharpe is the number that matters; maxDD the one you feel."""
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


def timing_weights(close: pd.Series, fast: int = 50, slow: int = 200, allow_short: bool = False) -> pd.Series:
    """Position from the crossover, set at yesterday's close (lagged): long in golden, cash/short else."""
    state = cross_state(close, fast, slow)
    w = (state > 0).astype(float)
    if allow_short:
        w = state.clip(-1, 1).astype(float)
    return w.shift(1).rename("weight")


def timing_returns(close: pd.Series, cost_bps: float = 2.0, fast: int = 50, slow: int = 200,
                   allow_short: bool = False) -> pd.Series:
    """The crossover timing stream, net of ``cost_bps`` charged on each change of position."""
    w = timing_weights(close, fast, slow, allow_short)
    r = close.pct_change()
    gross = w * r
    cost = (cost_bps * 1e-4) * w.diff().abs()
    return (gross - cost).dropna().rename("timing")


def buy_hold(close: pd.Series) -> pd.Series:
    return close.pct_change().rename("buy_hold")


def turnover_ann(close: pd.Series, **kw) -> float:
    """Annualised turnover — low (a 50/200 cross flips only a few times a year), but each flip costs."""
    w = timing_weights(close, **kw)
    return float(w.diff().abs().mean() * TRADING_DAYS_PER_YEAR)


def compare(close: pd.Series, cost_bps: float = 2.0, periods_per_year: int = TRADING_DAYS_PER_YEAR,
            **kw) -> dict:
    """Crossover timing (net) vs buy-and-hold over the matched (post-warmup) window."""
    t = timing_returns(close, cost_bps=cost_bps, **kw)
    bh = buy_hold(close).reindex(t.index)
    s_t, s_b = summary(t, periods_per_year), summary(bh, periods_per_year)
    return {
        "timing": s_t,
        "buy_hold": s_b,
        "sharpe_gain": float(s_t["sharpe"] - s_b["sharpe"]),
        "avg_exposure": float(timing_weights(close, **kw).reindex(t.index).mean()),
        "turnover_ann": float(turnover_ann(close, **kw)),
        "n_days": int(s_t["n_days"]),
    }


def cost_sweep(close: pd.Series, roundtrip_bps=(0, 2, 5, 10, 20), periods_per_year=TRADING_DAYS_PER_YEAR, **kw):
    """Net crossover Sharpe minus buy-and-hold Sharpe as cost rises."""
    rows = {}
    for c in roundtrip_bps:
        cmp = compare(close, cost_bps=c, periods_per_year=periods_per_year, **kw)
        rows[c] = {"timing_sharpe": cmp["timing"]["sharpe"], "sharpe_gain": cmp["sharpe_gain"]}
    out = pd.DataFrame(rows).T
    out.index.name = "cost_bps"
    return out
