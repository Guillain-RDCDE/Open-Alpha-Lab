"""Study 429 — Mass Index (Donald Dorsey, 1992).

The "reversal bulge": when the Mass Index (a 25-day sum of the single/double EMA
ratio of the daily high-low range) rises above 27 and then falls back below 26.5, a
trend reversal is supposedly imminent. This package exposes a cache-first data layer
(:mod:`mass_index.data`) and the detector + event study + fade-timing rule + honest
arbiters (:mod:`mass_index.strategy`).
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
