"""Costs and the break-even — the whole tradability question for a daily-rebalanced equity book.

Cross-sectional reversal is real gross of costs, but it is the canonical *liquidity-provision*
strategy: its return is the rent paid to whoever absorbs the crowd's short-term overreaction, and that
rent is collected one day at a time with near-total turnover. So the headline number isn't the gross
Sharpe — it's the **break-even cost**: the per-unit cost at which the net edge hits zero. If that sits
below realistic equity trading costs (spread + impact + commission, ~5–10 bp round-trip on average
S&P names), the edge is a `MIRAGE`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import book_returns, reversal_signal, summary, turnover


def cost_sweep(returns: pd.DataFrame, roundtrip_bps=(0, 2, 5, 10, 20), **kw) -> pd.DataFrame:
    """Net Sharpe / CAGR of the reversal book as the per-trade cost rises — where the edge dies."""
    rows = {}
    for c in roundtrip_bps:
        s = summary(book_returns(returns, cost_bps=c, **kw))
        rows[c] = {"sharpe": s["sharpe"], "cagr": s["cagr"]}
    out = pd.DataFrame(rows).T
    out.index.name = "cost_bps"
    return out


def breakeven_cost_bps(returns: pd.DataFrame, hi: float = 50.0, **kw) -> float:
    """Per-unit cost (bps) at which the book's mean net return crosses zero: ``gross_mean /
    (turnover · 1e-4)``. Returns 0.0 if the book doesn't make money gross."""
    gross_mean = book_returns(returns, cost_bps=0.0, **kw).mean()
    tpd = turnover(returns, **kw)
    if tpd <= 0 or gross_mean <= 0:
        return 0.0
    return float(min(hi, gross_mean / (tpd * 1e-4)))
