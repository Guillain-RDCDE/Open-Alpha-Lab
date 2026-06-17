"""Study 243 — Graham NCAV: price below net current asset value as a large-cap screen.

NCAV = CurrentAssets - TotalLiabilities.
Graham's original rule: buy when price < 2/3 * (NCAV / shares).

Built from the desk's shared EDGAR caches; market cap from yfinance monthly prices.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
