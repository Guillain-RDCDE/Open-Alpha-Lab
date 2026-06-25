"""Study 500 — Polarity-Flip (role reversal: old resistance becomes support).

A mechanical, falsifiable encoding of the chartists' "polarity principle": once price
**breaks above** a prior swing-high resistance level, that broken level is supposed to
*flip role* and act as **support** on the first pullback — so the first retest of a freshly
broken resistance is a high-probability **buy** (it should bounce). We test that claim as a
forward-return study against a drift-matched random-entry baseline and a level-scramble
placebo, with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
