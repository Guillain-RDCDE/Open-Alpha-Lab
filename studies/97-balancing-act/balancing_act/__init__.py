"""Study 97 — Balancing-Act: the classic fixed 60/40 stock/bond portfolio vs 100% stocks.

The fixed 60% SPY / 40% Treasuries blend, rebalanced annually — distinct from Study 68
(All-Weather / risk parity), which is *volatility*-weighted. Here the weights are fixed.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
