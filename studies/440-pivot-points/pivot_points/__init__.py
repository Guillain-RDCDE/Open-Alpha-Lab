"""Study 440 — Floor-Trader Pivot Points.

Do the classic floor-trader pivots (P, R1/R2, S1/S2 from the *prior* session's H/L/C)
act as real intraday support/resistance — i.e. does price *respect* the level (bounce /
reverse on a touch) more than it respects a **randomly placed** control level on the same
tape? The engine plus its honest controls live in :mod:`pivot_points.strategy`; the data
layer (synthetic planted-edge tape + cached yfinance intraday bars) in
:mod:`pivot_points.data`.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
