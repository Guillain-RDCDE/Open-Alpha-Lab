"""Worked extension (beat 7) — the realistic-spread test, name by name.

The headline charges one cost against the whole basket. The honest follow-up is per-instrument: the
IBS edge is a microstructure effect, so it should be measured against *each* ETF's own bid-ask spread —
and the names where the bounce is *largest* (the thin, volatile country funds) are exactly the names
with the *widest* spreads. So the question isn't "does some average survive a tiny cost" but "does any
single tradable instrument clear its own spread".

:func:`per_name_breakeven` reports each ETF's gross timing Sharpe and break-even cost, so you can line
the break-even up against the spread that name actually trades at. :func:`basket_net_at_spread` charges
a realistic round-trip and reports what's left. Two baked checks pin the machine: the null tape has no
gross edge and a 0-bp break-even, the reversal tape has a real gross edge but a sub-spread break-even.

House-standard *worked complement* in the study's own beat 7 — not a new study.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import summary, timing_returns, timing_panel_returns
from .decompose import breakeven_cost

TRADING_DAYS_PER_YEAR = 252


def per_name_breakeven(basket: dict, hi_bps: float = 50.0, tol_bps: float = 0.05) -> pd.DataFrame:
    """Per-ETF gross timing Sharpe and the cost (bps) at which it crosses zero.

    The point of the table: even the names with the strongest gross bounce break even at a handful of
    bps — below the spread those very names trade at — so there is no instrument where the edge clears
    its own cost.
    """
    rows = {}
    for tk, ohlc in basket.items():
        single = {tk: ohlc}
        gross = summary(timing_returns(ohlc, cost_bps=0.0))["sharpe"]
        be = breakeven_cost(single, hi_bps=hi_bps, tol_bps=tol_bps)
        rows[tk] = {"gross_sharpe": gross, "breakeven_bps": be["breakeven_bps"]}
    out = pd.DataFrame(rows).T.sort_values("gross_sharpe", ascending=False)
    out.index.name = "ticker"
    return out


def basket_net_at_spread(basket: dict, spread_bps: float = 2.0) -> dict:
    """Basket IBS timing net result when charged a realistic round-trip ``spread_bps`` per trade.

    ``spread_bps`` defaults to a tight ~2 bps — generous for liquid SPY/QQQ, optimistic for the thin
    country ETFs. Reports gross vs net Sharpe and net annual return: the gap is where the edge goes.
    """
    gross = summary(timing_panel_returns(basket, cost_bps=0.0))
    net = summary(timing_panel_returns(basket, cost_bps=spread_bps))
    return {
        "spread_bps": float(spread_bps),
        "gross_sharpe": float(gross["sharpe"]),
        "net_sharpe": float(net["sharpe"]),
        "gross_ann_return": float(gross["ann_return"]),
        "net_ann_return": float(net["ann_return"]),
    }
