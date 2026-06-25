"""Study 459 — Hikkake pattern (the false-breakout trap).

A mechanical, falsifiable encoding of Daniel Chesler's hikkake: an *inside bar*, then a
*false breakout* one way, then a snap-back *through* the inside-bar range — a trap that
supposedly forecasts a move in the reversal direction (long after a failed downside break,
short after a failed upside break). We test that as a direction-signed forward-return study
against an exposure-matched random-entry baseline and a scrambled-direction placebo, with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
