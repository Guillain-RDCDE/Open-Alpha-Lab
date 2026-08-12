"""Study 871 — The Rank Effect.

Hartzmark (2015): investors disproportionately **sell the best- and worst-ranked
positions** in their portfolio, so the top- and bottom-ranked names carry predictable
selling pressure. We rank a liquid US cross-section by trailing return each day and test
whether the **extreme-ranked** names under-earn the **middle** of the ranking next
period — a rank-extremity short — while **controlling for the raw trailing-return level**.

* ``data``     — the real cross-section (yfinance daily OHLC, cached under the study's
                 own ``_cache/`` through the ``quantlab.universe`` survivorship guard)
                 plus a deterministic seeded synthetic positive control (a planted
                 rank-extremity penalty, null at ``edge=0``).
* ``strategy`` — the trailing-return rank, the extremity score, the long-middle /
                 short-extremes spread, a level-controlled Fama-MacBeth extremity slope,
                 the inference primitives (Welch / one-sample / Newey-West HAC / Wilson /
                 placebo), and the costed timer.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
