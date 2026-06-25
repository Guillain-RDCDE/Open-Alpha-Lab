"""Study 472 — WaveTrend (LazyBear) oscillator cross.

A mechanical, falsifiable encoding of LazyBear's *WaveTrend* oscillator: an
EMA of the typical price is centred and normalised by its mean-absolute
deviation, then smoothed twice (WT1) with a 4-period SMA signal line (WT2).
The folklore says a WT1 cross **up** through WT2 while in **oversold**
territory is a high-probability buy. We test that as a forward-return study
against a drift-matched random-entry baseline and a parameter-scramble
placebo, with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
