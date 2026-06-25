"""Study 480 — Darvas Box (box breakout).

A mechanical, falsifiable encoding of Nicolas Darvas' box theory: a new high starts a
*box*; the box top is the recent high, the box bottom the subsequent consolidation low.
The folklore says a **close above the box top** is a high-probability long (the breakout
forecasts a continuation), entered the next close with an ATR/box stop. We test that as a
forward-return study against a drift-matched random-entry baseline and a shuffled-box
placebo, with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
