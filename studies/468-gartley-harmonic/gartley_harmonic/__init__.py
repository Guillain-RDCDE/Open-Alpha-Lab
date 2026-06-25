"""Study 468 — Gartley / AB=CD Harmonic patterns (Fibonacci-ratio reversals).

A mechanical, falsifiable encoding of harmonic-pattern trading. An XABCD swing is a
five-point zig-zag (X-A-B-C-D) of confirmed pivots; the folklore says that when the swings'
*retracement ratios* land on the Fibonacci grid (0.618 / 0.786 / 1.272 / 1.618 within a
tolerance), point **D** marks a reversal — a high-probability long in an up-context. We test
that as a forward-return study against a drift-matched random-entry baseline and a
ratio-scrambling placebo, with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
