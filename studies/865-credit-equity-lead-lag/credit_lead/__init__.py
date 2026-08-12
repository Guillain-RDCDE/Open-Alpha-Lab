"""Study 865 — Credit → Equity Lead-Lag.

"Credit leads equity": does the trailing 1-4-week **duration-hedged high-yield credit**
return (HYG in excess of IEF) **predict the next week's SPY return** (a Granger-style
lead-lag)? Tested as a weekly predictive regression with a Newey-West *t* on the slope, a
companion risk-on/off discrimination, a permutation placebo, a two-era cut, a costed
SPY↔IEF timing overlay vs a 100%-SPY buy-and-hold, and a seeded synthetic positive control.
"""
from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
