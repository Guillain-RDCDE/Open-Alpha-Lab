"""Study 498 — Dual Thrust (opening-range breakout).

A mechanical, falsifiable encoding of Michael Chalek's Dual Thrust system: from an
``N``-day high/low/close span, form a *Range* and two trigger bands around today's open
— ``buy_line = open + k1*Range`` and ``sell_line = open - k2*Range``. A long fires when
price breaks above the buy line; a short when it breaks below the sell line. The folklore
says the opening-range breakout "catches the day's trend." We test that as a forward-return
study against a drift-matched random-entry baseline and a parameter-scramble placebo, with
costs — the only honest test on an upward-drifting tape.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
