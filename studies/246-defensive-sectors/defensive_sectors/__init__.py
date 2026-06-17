"""Study 246 — Defensive-Sectors Leadership.

When defensive sectors (XLP staples + XLU utilities) outperform the broad
market (SPY) on a combined relative-strength basis, the folk wisdom reads it
as a risk-off warning: institutional money is rotating into safety, and a
drawdown is coming.  This package tests whether the combined XLP+XLU/SPY
relative strength -- and its momentum -- actually *forecasts* forward SPY
returns or whether it is merely coincident with stress.

Package layout
--------------
- ``data``      -- synthetic generator + real Yahoo daily loader (XLP, XLU, SPY).
- ``strategy``  -- signal construction, quintile forecasting tests, timing overlay,
                   regime stats, summarize helpers.
"""

from . import data, strategy  # noqa: F401

__version__ = "0.1.0"
