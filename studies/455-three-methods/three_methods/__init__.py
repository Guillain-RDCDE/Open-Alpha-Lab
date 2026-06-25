"""Study 455 — Rising/Falling Three Methods (candlestick continuation).

A mechanical, falsifiable encoding of the classic Japanese continuation pattern:
a long candle, three small counter-candles held *inside* its range, then a long
candle closing past the first — read as "the trend pauses, then resumes." We test
that continuation claim as a forward-return study against a drift-matched
random-entry baseline and a shuffled-body placebo, with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
