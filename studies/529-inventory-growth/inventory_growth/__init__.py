"""Study 529 — Inventory-Growth (Belo–Lin 2012; Thomas–Zhang 2002).

The inventory-growth anomaly: firms that grow inventory aggressively (scaled by
total assets) earn *low* future returns — a real-investment / over-extrapolation
story. We sort a survivor basket of inventory-heavy large caps on annual inventory
growth, go long the low-growth quintile and short the high-growth quintile, and
test whether the spread is real on the tape (it is not, on this thin survivor panel).
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
