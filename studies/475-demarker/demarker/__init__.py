"""Study 475 — DeMarker (DeMark's exhaustion oscillator).

A mechanical, falsifiable encoding of Thomas DeMark's DeMarker indicator: a bounded
0-1 oscillator built from smoothed up-moves over up+down moves of the *highs and lows*.
The folklore says a reading below 0.3 marks an oversold *exhaustion* turn — so when the
DeMarker rises out of <0.3, price is "about to reverse up". We test that as a
forward-return study against a drift-matched random-entry baseline and a phase-scramble
placebo, with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
