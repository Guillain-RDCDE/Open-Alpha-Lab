"""Study 489 — Chaikin Oscillator (A/D-line momentum).

A mechanical, falsifiable encoding of Marc Chaikin's oscillator. The Accumulation/
Distribution Line (ADL) cumulates each bar's volume weighted by the *Money Flow
Multiplier* ((close-low)-(high-close))/(high-low); the Chaikin Oscillator is then
EMA3(ADL) - EMA10(ADL). The folklore says A/D momentum *leads price*: when the
oscillator crosses **above zero**, accumulation is gathering and price is about to
rise. We test that as a forward-return study against a drift-matched random-entry
baseline and a volume/MFM-scramble placebo, with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
