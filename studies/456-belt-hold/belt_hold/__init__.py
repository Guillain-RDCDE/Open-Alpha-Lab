"""Study 456 — Belt-Hold (opening marubozu).

A mechanical, falsifiable encoding of the bullish belt-hold candlestick: a session
that **opens at its low** (no lower wick) and **closes well up** after a downtrend.
The folklore says the open-at-the-extreme marks a reversal — buyers seize control
from the first tick and the prior down-move turns. We test that as a forward-return
study against a drift-matched random-entry baseline and a shape-scramble placebo,
with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
