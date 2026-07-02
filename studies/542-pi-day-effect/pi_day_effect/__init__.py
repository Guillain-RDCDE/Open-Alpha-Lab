"""Study 542 — Pi-Day-Effect.

A tongue-in-cheek numerology anomaly: do returns on 'mathematical constant' dates
(Pi Day 3/14, Euler's day 2/7, Tau day 6/28, ...) differ from ordinary trading days?

- ``data.py``     : deterministic synthetic daily series (single knob plants a Pi-Day bump)
                    + a cache-first real SPY/GSPC tape (fetch=False by default).
- ``strategy.py`` : the two-sample test on constant-days vs the rest, a random-date-set
                    placebo null, a multiple-testing-aware sweep across constant dates,
                    a cost model for a naive calendar timing rule, a multi-window
                    robustness sweep, and a seed-robust synthetic positive control.
"""
