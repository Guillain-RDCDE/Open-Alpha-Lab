"""Study 1015 — The 1929 Overlay.

The recurring viral chart lays today's index over 1929 (or 1987, or 2000), prints a correlation
above 0.9 and lets the reader draw the conclusion. This study turns the chart into a forecaster
that can be scored — find the past windows whose path best matches today's, forecast what
followed them — runs every reasonable version of it out of sample on 92 years of monthly total
returns and 33 years of daily S&P 500 prices, and then asks the question the chart never asks:
how high a correlation would a random walk with the same drift and volatility have printed?

- :mod:`overlay.data` — the two real tapes (``quantlab.bundled``, SHA-pinned, offline) and the
  deterministic synthetic world with a template that genuinely recurs.
- :mod:`overlay.strategy` — the analogue search, the forecast, the overlap-robust scoring, the
  full sweep with multiple-testing corrections, the random-walk null and the pre-registered
  verdict.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
