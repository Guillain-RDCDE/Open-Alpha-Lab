"""Study 352 — Opening-Range Breakout (the viral 5-minute ORB).

Tests the day-trading-TikTok staple (steelmanned via Zarattini & Grewal 2023, "Can Day
Trading Really Be Profitable?"): mark the high/low of the first 5 (or 15) minutes, go long on a
break above / short below, exit at the close. The honest question is not "does it make money"
but "does breaking the opening range beat a **same-exposure random entry** on the same days,
after realistic intraday costs?"

See :mod:`opening_range_breakout.data` (yfinance 5m real tape + a deterministic synthetic
positive control with planted intraday trend-persistence) and
:mod:`opening_range_breakout.strategy` (the ORB rule, the random-entry null, the paired t-stat,
and the one-way cost sweep)."""

from . import data, strategy

__all__ = ["data", "strategy"]
