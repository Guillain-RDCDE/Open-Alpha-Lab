"""Study 497 — Woodie's Pivot Points (close-weighted floor-trader pivots).

A mechanical, falsifiable encoding of Woodie's pivots. Where the classic floor-trader
pivot is P = (H+L+C)/3, Woodie's variant *double-weights the close*: P = (H+L+2C)/4,
with R1/S1/R2/S2 derived from it. The folklore says yesterday's Woodie **S1** is an
intraday support: when today's price reaches down to S1 it should bounce. We test that
as a forward-return study against a drift-matched random-entry baseline and a
random-level placebo, with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
