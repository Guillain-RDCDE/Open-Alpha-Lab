"""Study 177 — Megacap-Concentration: does 'just buy the biggest' beat the index?

The Magnificent-Seven era made concentrating in a handful of megacaps look like genius.
But is it recency bias? We test top-N (N=7 or 10) concentration across the full EDGAR
history (2007–2025) vs SPY and equal-weight, split by decade.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
