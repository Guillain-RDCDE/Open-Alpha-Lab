"""Study 408 — Three Black Crows candlestick pattern.

A precise OHLC detector for the three-black-crows reversal candle, an event study of the
forward 1/3/5/10-day return after each occurrence vs the unconditional base rate, honest
arbiters (HAC / one-sample t, a label-shuffle placebo, costs), and a deterministic
synthetic positive control with a planted-edge knob.
"""

from . import data, strategy  # noqa: F401
