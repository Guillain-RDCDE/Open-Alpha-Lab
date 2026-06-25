"""Study 466 — Diamond Top (broadening-then-narrowing reversal).

A mechanical, falsifiable encoding of the classic "diamond top" chart pattern: a
rising market whose swing range first **broadens** (a megaphone — higher highs and
lower lows) and then **narrows** (a symmetrical triangle — lower highs and higher
lows), forming a diamond. The folklore says this rare shape **calls the turn**: when
price breaks **down** out of the diamond, sell/short. We encode that as a forward-return
study against a drift-matched random-entry baseline and a shuffled-geometry placebo, with
costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
