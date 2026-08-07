"""Study 829 — Global Sovereign-Bond Momentum.

Time-series (12-minus-1-month) momentum on a small panel of foreign / global
sovereign-bond ETFs. ``data`` builds the real month-end total-return tape (yfinance,
cached) and the seeded synthetic control; ``strategy`` implements the trend signal, the
costed backtest, and the shared inference primitives.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
