"""Study 454 — Three-Outside-Up / Three-Outside-Down (engulf + confirm).

A mechanical, falsifiable encoding of the *three-outside* candlestick pattern. The setup is a
two-bar **engulfing** candle (a small body fully swallowed by the next, opposite-coloured body)
followed by a **third confirming** candle that closes further in the engulf direction. The
folklore says the confirmation turns the engulfing into a high-probability reversal: long on a
three-outside-**up**, short on a three-outside-**down**. We test that as a forward-return study
against a drift-matched random-entry baseline and a body-shuffle placebo, with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
