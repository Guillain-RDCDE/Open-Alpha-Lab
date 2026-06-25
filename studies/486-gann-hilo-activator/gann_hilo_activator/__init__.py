"""Study 486 — Gann Hi-Lo Activator (a trailing flip line).

A mechanical, falsifiable encoding of the *Gann Hi-Lo Activator*: a trailing stop-and-reverse
line built from a simple moving average of recent **highs** and **lows**. When price closes
above the activator the line "flips" to track the SMA of lows (long regime); when price closes
below it, the line flips to track the SMA of highs (short regime). The folklore says the **flip
forecasts trend** — go long on a flip up, ride the move. We test that as a forward-return study
against a drift-matched random-entry baseline and a shuffled-flip placebo, with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
