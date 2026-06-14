"""Study 131 — Utilities-Canary.

When defensive utilities (XLU) outperform the broad market (SPY) on a relative-
strength basis, the folk wisdom reads it as a risk-off warning: "the canary in the
coal mine is singing, a drawdown is coming."  This package tests whether XLU/SPY
relative strength — and its momentum — actually forecasts forward SPY returns or
drawdowns, versus an unconditional baseline.

Package layout
--------------
- ``data``      — synthetic generator + real Yahoo daily loader.
- ``strategy``  — signal construction, forecasting tests, summarize helpers.
"""

from . import data, strategy  # noqa: F401

__version__ = "0.1.0"
