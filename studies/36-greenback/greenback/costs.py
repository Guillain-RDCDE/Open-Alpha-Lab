"""Costs and the break-even — the tradability side of the carry / momentum / combo books.

Carry rebalances slowly (rate differentials crawl), so its turnover is *low* and cost is not the threat —
the crash is. Momentum turns over faster (12-month trend, monthly refresh). The combo sits between. The
honest cost question per book: at what per-unit cost does the net edge cross zero (the **break-even**),
and how does the Sharpe decay as costs rise?
"""

from __future__ import annotations

import pandas as pd

from .strategy import (carry_returns, carry_weights, combine, momentum_returns, momentum_weights,
                       summary, turnover_ann)


def cost_sweep(xret: pd.DataFrame, rate_diffs: pd.DataFrame, roundtrip_bps=(0, 10, 25, 50, 100),
               which: str = "combo", **kw) -> pd.DataFrame:
    """Net Sharpe / annual return of the chosen book (``carry`` / ``momentum`` / ``combo``) vs cost."""
    rows = {}
    for c in roundtrip_bps:
        if which == "carry":
            r = carry_returns(xret, rate_diffs, cost_bps=c, **kw)
        elif which == "momentum":
            r = momentum_returns(xret, cost_bps=c, **kw)
        else:
            r = combine(carry_returns(xret, rate_diffs, cost_bps=c),
                        momentum_returns(xret, cost_bps=c))
        s = summary(r)
        rows[c] = {"sharpe": s["sharpe"], "ann_return": s["ann_return"]}
    out = pd.DataFrame(rows).T
    out.index.name = "cost_bps"
    return out


def breakeven_cost_bps(xret: pd.DataFrame, rate_diffs: pd.DataFrame, which: str = "carry",
                       hi: float = 500.0, **kw) -> float:
    """Per-unit cost (bps) at which the book's mean net return crosses zero: ``gross_mean / (turnover·1e-4)``.

    Returns 0.0 if the book doesn't make money gross. Uses the *unscaled* book so turnover is well-defined.
    """
    if which == "momentum":
        gross_mean = momentum_returns(xret, cost_bps=0.0, target_vol_ann=None, **kw).mean()
        tover = turnover_ann(momentum_weights(xret))
    else:
        gross_mean = carry_returns(xret, rate_diffs, cost_bps=0.0, target_vol_ann=None, **kw).mean()
        tover = turnover_ann(carry_weights(rate_diffs))
    if gross_mean <= 0:
        return 0.0
    if tover <= 0:
        return float(hi)                              # zero turnover ⇒ cost never bites (carry's fixed rates)
    return float(min(hi, gross_mean / (tover * 1e-4)))
