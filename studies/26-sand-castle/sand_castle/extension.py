"""Worked extension (beat 7) — does shrinking the covariance rescue the optimizer?

The textbook fix for an unstable ``C^{-1}`` is **shrinkage** (Ledoit & Wolf 2004): blend the noisy sample
covariance toward a stable target (here its diagonal). Shrinkage tames the extreme weights, so the
question is whether it lets the optimizer finally beat the naive portfolio net of cost. The punchline,
measured here: shrinkage helps the optimizer climb back *toward* the naive book — and at full shrink the
two converge by construction (a diagonal ``C`` makes ``C^{-1}E`` proportional to ``E``). So the best the
optimization can do is *stop optimizing*; the covariance inversion never adds tradable value.

:func:`shrink_sweep` traces optimized vs naive net Sharpe across shrinkage intensities.

House-standard *worked complement* in the study's own beat 7 — not a new study.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import statarb_returns, summary

TRADING_DAYS_PER_YEAR = 252


def shrink_sweep(panel: pd.DataFrame, market: pd.Series | None = None, cost_bps: float = 5.0,
                 shrinks=(0.0, 0.3, 0.6, 0.9, 1.0), periods_per_year: int = TRADING_DAYS_PER_YEAR, **kw) -> pd.DataFrame:
    """Optimized net Sharpe across covariance-shrinkage intensities, vs the (shrink-free) naive book.

    ``shrink = 0`` is the raw inverse; ``shrink = 1`` is a diagonal covariance (≈ the naive book). The
    naive net Sharpe is the constant reference. A real benefit from optimization would show the optimized
    curve *above* the naive line at some intermediate shrink; the study's finding is that it doesn't.
    """
    naive = summary(statarb_returns(panel, market, optimized=False, cost_bps=cost_bps, **kw), periods_per_year)["sharpe"]
    rows = {}
    for s in shrinks:
        opt = summary(statarb_returns(panel, market, optimized=True, shrink=s, cost_bps=cost_bps, **kw), periods_per_year)["sharpe"]
        rows[s] = {"optimized_net_sharpe": opt, "naive_net_sharpe": naive}
    out = pd.DataFrame(rows).T
    out.index.name = "shrink"
    return out
