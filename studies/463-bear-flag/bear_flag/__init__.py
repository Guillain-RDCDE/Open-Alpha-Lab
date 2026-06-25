"""Study 463 — Bear-Flag (continuation after the pole).

A mechanical, falsifiable encoding of the classic **bear flag**: a sharp drop (the
*pole*), then a small *up-sloping* consolidation (the *flag*) that drifts against the
move on lighter range, followed by a **breakdown** that "continues the drop". The
folklore says the flag *forecasts continuation* — short the breakdown and ride the
second leg down. We test that as a forward-return study against a drift-matched
random-entry baseline and a flag-geometry placebo, with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
