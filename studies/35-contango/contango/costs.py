"""Costs and the break-even — the (gentler) tradability question for a slow commodity carry book.

Unlike a daily-rebalanced equity book (Study 33), commodity carry rolls on a slow term-structure signal,
so it turns over modestly and the *cost* question is mild — the real tradability risk is the **crash tail**
(measured in the notebooks), not transaction costs. Still, we report the break-even cost: the per-unit
cost at which the net edge hits zero. For commodity carry this should sit comfortably above realistic
futures round-trip costs (~2–5 bp on liquid contracts), so costs are not the binding constraint.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import book_returns, summary, turnover


def cost_sweep(returns: pd.DataFrame, roll_yield: pd.DataFrame, roundtrip_bps=(0, 2, 5, 10, 20), **kw) -> pd.DataFrame:
    """Net Sharpe / CAGR of the carry book as the per-trade cost rises."""
    rows = {}
    for c in roundtrip_bps:
        s = summary(book_returns(returns, roll_yield, cost_bps=c, **kw))
        rows[c] = {"sharpe": s["sharpe"], "cagr": s["cagr"]}
    out = pd.DataFrame(rows).T
    out.index.name = "cost_bps"
    return out


def breakeven_cost_bps(returns: pd.DataFrame, roll_yield: pd.DataFrame, hi: float = 200.0, **kw) -> float:
    """Per-unit cost (bps) at which the book's mean net return crosses zero: ``gross_mean /
    (turnover · 1e-4)``. Returns 0.0 if the book doesn't make money gross. For a slow carry book this is
    high (low turnover), so costs are not the binding constraint — the crash tail is."""
    gross_mean = book_returns(returns, roll_yield, cost_bps=0.0, **kw).mean()
    tpd = turnover(roll_yield)
    if tpd <= 0 or gross_mean <= 0:
        return 0.0
    return float(min(hi, gross_mean / (tpd * 1e-4)))
