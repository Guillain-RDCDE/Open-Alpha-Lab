"""Costs and the break-even — for the walk-forward (honest) book.

Even *if* the walk-forward net had a positive edge, a daily-direction strategy flips position often, so
transaction costs bite. The cost sweep shows the net Sharpe of the out-of-sample book as the per-trade
cost rises, and the break-even cost is the per-unit cost at which the net edge hits zero. For an
overfit black box the gross OOS edge is already ~0, so the break-even is at or below zero — there is
nothing for costs to erode.
"""

from __future__ import annotations

import pandas as pd

from .strategy import book_returns, summary


def cost_sweep(close: pd.Series, positions: pd.Series, roundtrip_bps=(0, 5, 10, 20, 50)) -> pd.DataFrame:
    """Net Sharpe / CAGR of the walk-forward book as the per-trade cost rises."""
    rows = {}
    for c in roundtrip_bps:
        s = summary(book_returns(close, positions, cost_bps=c))
        rows[c] = {"sharpe": s["sharpe"], "cagr": s["cagr"]}
    out = pd.DataFrame(rows).T
    out.index.name = "cost_bps"
    return out


def turnover(positions: pd.Series) -> float:
    """Average daily one-way turnover (mean |Δposition|) of the +1/−1 book — how often it flips."""
    pos = pd.Series(positions).astype(float)
    return float(pos.diff().abs().fillna(pos.abs()).mean())


def breakeven_cost_bps(close: pd.Series, positions: pd.Series, hi: float = 100.0) -> float:
    """Per-unit cost (bps) at which the book's mean net return crosses zero: ``gross_mean /
    (turnover · 1e-4)``. Returns 0.0 if the book doesn't make money gross — the usual black-box outcome."""
    gross_mean = book_returns(close, positions, cost_bps=0.0).mean()
    tpd = turnover(positions)
    if tpd <= 0 or gross_mean <= 0:
        return 0.0
    return float(min(hi, gross_mean / (tpd * 1e-4)))
