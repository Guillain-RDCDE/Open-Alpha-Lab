"""Study 189 — Double-Top / Double-Bottom (M & W reversal patterns).

The double-top (M) is two price peaks at roughly the same level separated by a
trough; the double-bottom (W) is the mirror.  The classic trading rule says:
*wait for the price to break below the trough (the "neckline") to confirm the
double-top, then short expecting a downward reversal* — and vice versa for the
double-bottom.

This package detects both patterns on daily OHLCV bars via :mod:`strategy`,
measures forward returns after confirmation, and pins the result against a
random-day placebo to find whether the pattern adds any information beyond
what the unconditional distribution already offers.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
