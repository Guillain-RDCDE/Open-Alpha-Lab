"""Study 509 -- Intermediate-Momentum: does momentum live in the t-12..t-7 window?

Novy-Marx (2012, "Is momentum really momentum?"): the cross-sectional momentum premium is
carried by the *intermediate* part of the formation window (twelve to seven months ago),
not the *recent* part (six to two months ago). We build long-shorts on both windows over a
large-cap survivor basket and contrast which one carries the drift -- with one execution
lag, a label-shuffle placebo, turnover-based costs + borrow, and a synthetic positive control.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
