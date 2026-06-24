"""Study 421 — Williams Alligator.

A trend-following indicator (Bill Williams, 1995) built from three smoothed moving
averages of the median price, each shifted forward in time:

    Jaw   = SMMA(median, 13) shifted 8 bars forward   (blue)
    Teeth = SMMA(median, 8)  shifted 5 bars forward   (red)
    Lips  = SMMA(median, 5)  shifted 3 bars forward   (green)

The folk story: when the three lines are intertwined the "alligator sleeps" (no trend,
stay flat); when they fan out in order (Lips > Teeth > Jaw, or the reverse) the
"alligator wakes and eats" — a trend is underway, ride it.

This package exposes the indicator + a long/flat (and long/short) timing rule and the
desk's honest arbiters (HAC t, a permutation placebo, costs, a benchmark race vs SMA(200)).
"""

from . import data, strategy

__all__ = ["data", "strategy"]
