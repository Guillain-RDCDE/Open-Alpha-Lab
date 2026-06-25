"""Study 499 — Trendline-Break.

A mechanical, falsifiable encoding of the chartist's "trendline break" signal: fit an
*uptrend line* by least-squares through the recent confirmed swing lows; the trend is
"intact" while price holds above the line, and a **confirmed close below the line** is the
classic break — exit the long / fire the short. The folklore says the break *forecasts* a
turn (momentum reverses once support cracks). We test that as a forward-return study against
a drift-matched random-entry baseline and a shuffled-slope geometry placebo, with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
