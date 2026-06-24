"""Study 450 — Andrews' Pitchfork (median-line channel).

A mechanical, falsifiable encoding of Alan Andrews' pitchfork: three swing pivots
P0, P1, P2 define a *median line* (P0 through the midpoint of P1-P2) and two parallel
*tines* through P1 and P2. The folklore says price "respects the channel" — it reverts
from the tines and gravitates to the median line. We test that as a forward-return
study against random-entry and shuffled-pivot placebos, with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
