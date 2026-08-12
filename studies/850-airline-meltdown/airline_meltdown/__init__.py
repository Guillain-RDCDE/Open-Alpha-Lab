"""Study 850 — Airline Operational Meltdown: does a very public operational collapse
dent the airline's own stock?

A curated table of the most infamous **operational meltdowns** of the modern era —
Delta's 2016 power-outage grounding and its 2024 CrowdStrike collapse, United's 2017
passenger-dragging PR crisis, the two Boeing 737-MAX grounding events (2019 fatal-crash
grounding, 2024 Alaska door-plug blowout), Southwest's October-2021 and Christmas-2022
cancellation collapses, American's 2021 Halloween meltdown, Spirit's 2021 cancellation
wave — is run through a single-name market-model event study on the *implicated*
carrier's stock (LUV / DAL / UAL / AAL / BA, benchmarked to SPY) and the one-month
drift that follows. Low-N by construction; usually **None**.

* ``data``     — the curated public-fact meltdown table (dates + implicated ticker +
                 cited source note), the real yfinance price tape (cached under the
                 study's own ``_cache/``), and a deterministic seeded synthetic control
                 with a TUNABLE planted event drop (null at ``edge=0``).
* ``strategy`` — the market-model abnormal-return event study (estimation-window
                 alpha/beta), the cross-event mean CAR at several horizons with its
                 one-sample *t*, a same-ticker permutation placebo over random
                 pseudo-event dates, the inference primitives (one-sample / Welch /
                 Newey-West HAC / Wilson / placebo), and the costed short-the-meltdown
                 timer.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
