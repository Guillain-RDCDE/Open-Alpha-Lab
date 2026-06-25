"""Study 461 — Descending Triangle (bearish continuation pattern).

A mechanical, falsifiable encoding of the classic *descending triangle*: a run of
**flat lows** (a horizontal support) under a series of **descending highs** (a falling
upper trendline), the two converging into an apex. The folklore says the pattern is a
bearish continuation that **breaks down as drawn** — price slices through the flat
support and keeps falling. We test that as a forward-return study: mechanically detect
the wedge of swing highs / flat lows, short on the close below support (entered the next
close), and race the short against a drift-matched random-entry baseline and a
scrambled-geometry placebo, with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
