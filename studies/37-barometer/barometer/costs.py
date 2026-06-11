"""Costs and the break-even — the tradability question for a slow, monthly cross-asset macro book.

Unlike the desk's daily-churn books (Slingshot, Rip-Tide), macro momentum is *cheap to run*: the signal
is the trend in monthly macro data, so the book turns over only a few times a year. Cost is therefore
**not** the threat that kills it — capacity, the slowness, and the episodic nature of the inflation hedge
are. Still, the desk charges costs against the alpha and reports the **break-even cost**: the per-unit
cost at which the net edge crosses zero. For a slow book that number is comfortably high.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import book_returns, inflation_tilt_weights, macro_momentum_weights, summary


def _turnover_per_month(macro: pd.DataFrame, kind: str, smooth: int, assets=None, order=None) -> float:
    w = macro_momentum_weights(macro, smooth=smooth, assets=assets, order=order) if kind == "macro_momentum" \
        else inflation_tilt_weights(macro, smooth=smooth, assets=assets, order=order)
    return float(w.diff().abs().sum(axis=1).mean())


def cost_sweep(returns: pd.DataFrame, macro: pd.DataFrame, kind: str = "macro_momentum",
               roundtrip_bps=(0, 5, 10, 25, 50), smooth: int = 3, assets=None, order=None) -> pd.DataFrame:
    """Net Sharpe / annual return of a macro book as the per-trade cost rises."""
    rows = {}
    for c in roundtrip_bps:
        s = summary(book_returns(returns, macro, kind=kind, cost_bps=c, smooth=smooth,
                                 assets=assets, order=order))
        rows[c] = {"sharpe": s["sharpe"], "ann_return": s["ann_return"]}
    out = pd.DataFrame(rows).T
    out.index.name = "cost_bps"
    return out


def breakeven_cost_bps(returns: pd.DataFrame, macro: pd.DataFrame, kind: str = "macro_momentum",
                       hi: float = 500.0, smooth: int = 3, assets=None, order=None) -> float:
    """Per-unit cost (bps) at which the book's mean net return crosses zero: ``gross_mean /
    (turnover_per_month · 1e-4)``. Returns 0.0 if the book makes nothing gross."""
    gross_mean = book_returns(returns, macro, kind=kind, cost_bps=0.0, smooth=smooth,
                              assets=assets, order=order).mean()
    tpm = _turnover_per_month(macro, kind, smooth, assets=assets, order=order)
    if tpm <= 0 or gross_mean <= 0:
        return 0.0
    return float(min(hi, gross_mean / (tpm * 1e-4)))
