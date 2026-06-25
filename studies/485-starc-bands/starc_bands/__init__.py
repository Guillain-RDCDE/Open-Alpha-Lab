"""Study 485 — STARC Bands (Stoller Average Range Channel).

A mechanical, falsifiable encoding of Manning Stoller's STARC bands: a short SMA of
the close flanked by an ATR envelope — ``upper = SMA + k·ATR``, ``lower = SMA − k·ATR``.
The folklore says a close *below the lower band* is a high-probability **buy** (price
mean-reverts back toward the SMA). We test that as a forward-return study against a
drift-matched random-entry baseline and a band-scrambling placebo, with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
