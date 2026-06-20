"""Study 322 — FOMC-Blackout.

Is the Fed's pre-meeting communications *blackout* — the ~10-day quiet period before
each FOMC decision when officials may not speak publicly — a "calm before the storm"
window that pays an excess drift?

This is a *different angle* from Study 135 (FOMC-Cycle). Study 135 replicates Cieslak,
Morse & Vuolteenaho's even/odd-week cycle, which is built around the *post*-meeting
inter-meeting calendar and concentrates on the days **after** each statement. Study 322
isolates the **pre-meeting blackout window itself** (the days leading up to the meeting),
classifies every trading day as blackout vs non-blackout, and asks whether the blackout
days carry an *excess* return (and lower volatility) over the rest of the year.

Public API:
    data.FOMC_DATES            — the scheduled FOMC decision dates, 1994-2026
    data.in_blackout           — boolean blackout flag for a DatetimeIndex
    data.synthetic_blackout    — deterministic offline tape with a tunable excess
    data.load_real             — cache-first SPY total-return tape, with blackout flag
    strategy.blackout_table    — blackout vs non-blackout means + HAC t + Welch diff
    strategy.placebo_shift     — calendar-shift placebo (move the window, re-measure)
    strategy.split_by_era      — pre/post-2011 contrast
    strategy.vol_table         — the "calm?" check (blackout vs non-blackout vol)
    strategy.cost_sweep        — a long-only-in-blackout timing book, net of costs
"""

from __future__ import annotations

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
