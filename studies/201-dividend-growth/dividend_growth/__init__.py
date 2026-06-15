"""Study 201 — Dividend-Growth.

Do stocks with *consistently rising* dividends outperform a broad equal-weight large-cap
basket and high-yield (non-growing) payers?  The idea is a staple of retail investing:
"dividend growers" signal quality, capital discipline, and management confidence, so they
should deliver superior risk-adjusted returns.

This package provides:

- :mod:`dividend_growth.data` — a deterministic synthetic annual panel (positive/null
  control) and a real loader that fetches yfinance ``.dividends`` + adjusted closes for a
  fixed basket of large-cap stocks, cached to ``_cache/`` as parquet.
- :mod:`dividend_growth.strategy` — the growth-screen logic (consecutive dividend raises
  over a configurable look-back), the equal-weight baseline, and the high-yield (non-growing)
  comparator; inference via HAC t-stat and bootstrap Sharpe CI reusing ``quantlab``.

Survivorship bias is named on the Signal axis: the basket is fixed to names alive today.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dividend_growth")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["data", "strategy"]
