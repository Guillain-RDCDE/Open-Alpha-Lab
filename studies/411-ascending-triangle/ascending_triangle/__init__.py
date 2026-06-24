"""Study 411 — Ascending Triangle (does the figure really break upward?).

Objective mechanical detector for the ascending-triangle figure (flat horizontal
resistance + a rising lower trendline) on daily OHLC, plus the breakout timing rule and
the desk's honest arbiters. See ``data`` (tapes + synthetic positive control) and
``strategy`` (detector + inference).
"""

from . import data, strategy  # noqa: F401
