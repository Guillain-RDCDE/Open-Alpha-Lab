"""Costs and benchmarks — why the reversion book's tradability question is the *opposite* of the trend
book's, and far less forgiving.

Study 31 (Trade-Winds) had an unusually friendly cost story: trend signals turn over slowly, so a few
basis points barely dent the Sharpe. The contrarian book is the mirror — a 1–5 day signal flips almost
daily, so turnover is an order of magnitude higher and the *same* per-unit cost compounds into a wall.
This module supplies:

  * the **cost sweep** — net Sharpe as the per-trade cost rises (where the gross edge meets the wall);
  * the **break-even cost** — the per-unit cost at which the net edge hits zero, the headline number;
  * the **long-only basket** benchmark (always-long, same vol-scaling) — to show that, net of costs,
    even doing *nothing* clever beats the frantic reversal book.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import TRADING_DAYS, book_returns, summary, turnover


def cost_sweep(returns: pd.DataFrame, roundtrip_bps=(0, 1, 2, 5, 10), **kw) -> pd.DataFrame:
    """Net Sharpe / CAGR of the contrarian book as the per-trade cost rises — does the edge survive?"""
    rows = {}
    for c in roundtrip_bps:
        s = summary(book_returns(returns, cost_bps=c, **kw))
        rows[c] = {"sharpe": s["sharpe"], "cagr": s["cagr"]}
    out = pd.DataFrame(rows).T
    out.index.name = "cost_bps"
    return out


def breakeven_cost_bps(returns: pd.DataFrame, hi: float = 20.0, **kw) -> float:
    """The per-unit cost (bps) at which the book's mean net return crosses zero.

    Net daily mean is linear in the cost (``gross_mean − cost·turnover``), so the break-even is just
    ``gross_mean / (turnover · 1e-4)``. Returns 0.0 if the book doesn't even make money gross."""
    gross_mean = book_returns(returns, cost_bps=0.0, **kw).mean()
    tpd = turnover(returns, **kw)
    if tpd <= 0 or gross_mean <= 0:
        return 0.0
    return float(min(hi, gross_mean / (tpd * 1e-4)))


def long_only_basket(returns: pd.DataFrame, vol_window: int = 63, target_vol: float = 0.10) -> pd.Series:
    """Always-long, equal-risk, vol-scaled basket of the SAME markets — the do-nothing-clever
    benchmark. If the contrarian book doesn't beat this net of costs, the *timing* destroyed value."""
    realised = returns.rolling(vol_window).std() * np.sqrt(TRADING_DAYS)
    per_mkt = target_vol / np.sqrt(max(1, returns.shape[1]))
    w = (per_mkt / realised).clip(upper=3.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return (w.shift(1) * returns).sum(axis=1).rename("long_only_basket")
