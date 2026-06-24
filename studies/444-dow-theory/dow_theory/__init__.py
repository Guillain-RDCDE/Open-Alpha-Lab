"""Study 444 — Dow Theory (Industrials / Transports confirmation).

Mechanical, falsifiable encoding of the oldest piece of US technical-analysis
folklore: a primary trend is only *valid* when the Dow Jones Industrial Average
and the Dow Jones Transportation Average **confirm** each other. We test the
tradable version — be long the Industrials only while the Transports confirm the
trend — against buy-and-hold, with a HAC t-stat, a label/regime placebo, and
realistic costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
