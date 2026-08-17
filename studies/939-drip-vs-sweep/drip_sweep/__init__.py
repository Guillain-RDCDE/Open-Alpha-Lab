"""Study 939 — DRIP or Sweep.

Reinvest each ETF distribution the day it lands, or sweep it into T-bills and
reinvest on a calendar schedule? Two accounting policies over the *same* holding,
raced on the reconstructed dividend stream of SPY, VYM and SCHD.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
