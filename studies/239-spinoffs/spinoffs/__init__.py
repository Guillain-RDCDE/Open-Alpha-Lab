"""Study 239 — Spinoffs: do spin-off children outrun the market?

Greenblatt (1997) and Cusatis, Miles & Woolridge (1993) document a post-spin-off
premium: the child entity earns abnormal returns vs the market in the first 1–2 years.
We test this on a hardcoded table of notable US spin-offs (2011–2024) using
yfinance daily prices and a SPY benchmark.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
