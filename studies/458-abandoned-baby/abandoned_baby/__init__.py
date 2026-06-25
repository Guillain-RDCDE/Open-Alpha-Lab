"""Study 458 — Abandoned-Baby (island doji reversal).

A mechanical, falsifiable encoding of the *bullish abandoned baby* candlestick pattern:
a three-bar island reversal where a **doji** (open ≈ close) gaps *down* away from the
prior down-candle and the *next* candle gaps back *up* away from the doji, leaving the
doji stranded on its own price island. The folklore says this island doji marks a turn —
buy the confirmation candle, ride the reversal up. We test that as a forward-return study
against a drift-matched random-entry baseline and a gap-scramble placebo, with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
