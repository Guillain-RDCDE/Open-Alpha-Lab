"""Costs and the break-even — the tradability question for the PEAD book.

PEAD is one of the most-documented anomalies, but its tradability has always been the catch: the drift is
small, it concentrates in small / illiquid names (Bernard-Thomas 1989; Chordia et al. 2009), and you pay
to roll the book as each name's surprise ages out. So the headline number isn't the gross Sharpe — it's
the **break-even cost**: the per-unit cost at which the net edge hits zero. If that sits below realistic
equity round-trip costs (spread + impact, worse in the small names where PEAD is strongest), the edge is
`FRAGILE` at best.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import book_returns, pead_weights, summary, turnover


def cost_sweep(panel: pd.DataFrame, events: pd.DataFrame, roundtrip_bps=(0, 2, 5, 10, 20), **kw
               ) -> pd.DataFrame:
    """Net Sharpe / CAGR of the PEAD book as the per-trade cost rises — where the edge dies."""
    rows = {}
    for c in roundtrip_bps:
        s = summary(book_returns(panel, events, cost_bps=c, **kw))
        rows[c] = {"sharpe": s["sharpe"], "cagr": s["cagr"]}
    out = pd.DataFrame(rows).T
    out.index.name = "cost_bps"
    return out


def breakeven_cost_bps(panel: pd.DataFrame, events: pd.DataFrame, hi: float = 200.0, **kw) -> float:
    """Per-unit cost (bps) at which the book's mean net return crosses zero: ``gross_mean /
    (turnover · 1e-4)``. Returns 0.0 if the book doesn't make money gross."""
    gross_mean = book_returns(panel, events, cost_bps=0.0, **kw).mean()
    tpd = turnover(panel, events, **kw)
    if tpd <= 0 or gross_mean <= 0:
        return 0.0
    return float(min(hi, gross_mean / (tpd * 1e-4)))
