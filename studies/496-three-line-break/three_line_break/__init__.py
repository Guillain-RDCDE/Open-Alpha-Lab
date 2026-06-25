"""Study 496 — Three-Line-Break (Sakata / Nison reversal chart).

A mechanical, falsifiable encoding of the classic Three-Line-Break (TLB) chart. TLB
draws a new "line" (block) only when the close pushes past the extreme of the prior
line; it **reverses** colour only after the close breaks the extremes of the **3**
most-recent opposite-coloured lines. The folklore says a TLB reversal **forecasts** a
new trend — go long on an up-line, flat on a reversal. We test that as a forward-return
study against a drift-matched random-entry baseline, a "shuffle the line-break geometry"
placebo, and a deterministic synthetic positive control, with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
