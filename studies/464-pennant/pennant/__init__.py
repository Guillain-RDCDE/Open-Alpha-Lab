"""Study 464 — Pennant (pole + symmetrical-triangle continuation).

A mechanical, falsifiable encoding of the classic *pennant* continuation pattern: a steep
near-vertical **pole** (a strong directional thrust) followed by a small **symmetrical
triangle** consolidation (converging swing highs and lows on shrinking range), then a
**breakout** in the pole's direction. The folklore says the pennant *continues the prior
thrust* — measure the forward move after the breakout. We test that as a forward-return study
against a drift-matched random-entry baseline and a pole-direction-scramble placebo, with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
