"""Study 457 — Kicker-Pattern (gap-and-reverse marubozu pair).

A mechanical, falsifiable encoding of the candlestick "kicker": two opposite
marubozu candles separated by a gap in the *new* direction. A bullish kicker is
a black (down) marubozu followed by a white (up) marubozu that **gaps up** above
the prior open; a bearish kicker is the mirror. The folklore says the kicker is
"one of the most reliable reversal signals" — a violent turn ignoring the prior
trend. We test that as a forward-return study, trading the kicker direction
entered at the next close, against a drift-matched random-entry baseline, a
gap-scramble placebo, and a deterministic synthetic positive control, with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
