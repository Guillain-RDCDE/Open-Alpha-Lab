"""Study 469 — Relative Vigor Index (RVI).

A mechanical, falsifiable encoding of John Ehlers' Relative Vigor Index: a momentum
oscillator built as a smoothed ratio of the bar's body (close − open) to its range
(high − low), with a symmetric-weighted four-bar smoother and a four-bar signal line.
The folklore says the RVI crossing **above** its signal line is a buy (vigor turning up),
the cross below a sell. We test that as a forward-return study against a drift-matched
random-entry baseline and a phase-scramble placebo, with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
