"""Study 534 — Revenue-Surprise-Drift (Jegadeesh & Livnat 2006).

Does the post-earnings drift follow the **revenue (sales) surprise**, not just the EPS
surprise? We build a clean event study on a fixed large-cap basket: per name we pull every
quarterly revenue figure from EDGAR (frame-tagged calendar quarters), form the **standardized
unexpected revenue** (SUR = the seasonal-random-walk revenue surprise, scaled by its own
rolling volatility), sort events into SUR quintiles, and measure the forward 1/5/20/60-day
drift of a top-minus-bottom long-short. Distinct from EPS-PEAD (study 363).
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
