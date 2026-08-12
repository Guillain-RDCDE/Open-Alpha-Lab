"""Study 869 — 52-Week-High Breakout Drift.

Distinct from George-Hwang **nearness** to the 52-week high (study 236): this study
tests the **EVENT** of a fresh 52-week-high *breakout*. When a name closes at a new
52-week high for the first time, does it **drift up** (breakout momentum) or **fade**
(resistance / anchoring)? We flag each name's new-52-week-high days point-in-time and
measure the forward 5- and 20-day return of a long-just-broke-out book versus the rest.

* ``data``     — the real cross-section (yfinance daily OHLC, cached under the study's
                 own ``_cache/`` through the ``quantlab.universe`` survivorship guard)
                 plus a deterministic seeded synthetic positive control (a planted
                 breakout->forward-drift relation, null at ``edge=0``).
* ``strategy`` — the point-in-time 52-week-high breakout flag, the forward-window event
                 sort (breakout book vs the rest), the inference primitives (Welch /
                 one-sample / Newey-West HAC / Wilson / placebo), and the costed timer.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
