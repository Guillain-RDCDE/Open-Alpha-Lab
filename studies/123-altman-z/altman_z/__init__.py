"""Study 123 — Altman-Z: the classic distress model as an equity signal.

Z = 1.2(WC/TA) + 1.4(RE/TA) + 3.3(EBIT/TA) + 0.6(MktEq/TotalLiab) + 1.0(Sales/TA).
Built from the desk's shared EDGAR caches; market cap from yfinance monthly prices.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
