"""Study 453 — Three-Inside-Up / Three-Inside-Down (Harami + confirmation).

A mechanical, falsifiable encoding of the classic three-bar candlestick reversal:
a bearish candle, an inside *harami* (a smaller candle contained in the first's body),
and a confirming third candle that closes back **past** the first candle's open — the
"three-inside-up" after a downtrend (mirror: three-inside-down after an uptrend). The
folklore says the confirmation candle "flips the trend". We test that as a forward-return
study against a random-entry baseline, with a placebo that drops the confirmation
requirement to ask the thesis question: *does the confirmation candle add edge?*
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
