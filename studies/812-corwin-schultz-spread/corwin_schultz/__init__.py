"""Study 812 — Corwin-Schultz Spread.

Corwin & Schultz (2012): the **bid-ask spread** can be estimated from the daily
**high** and **low** prices alone. Over any two consecutive days the observed high-low
range reflects both the true price variance and the *spread* (the high is transacted at
the ask, the low at the bid); the estimator separates the two. High estimated spread =
illiquid name = a candidate **illiquidity premium**, so a long high-spread / short
low-spread book is the classic Amihud-style illiquidity bet.

* ``data``     — the real cross-section (yfinance daily OHLC, cached under the study's
                 own ``_cache/`` through the ``quantlab.universe`` survivorship guard)
                 plus a deterministic seeded synthetic positive control (a planted
                 illiquidity-premium relation, null at ``edge=0``).
* ``strategy`` — the daily Corwin-Schultz high-low spread estimator, its trailing-month
                 average, the point-in-time cross-sectional sort (long high-spread /
                 short low-spread), the inference primitives (Welch / one-sample /
                 Newey-West HAC / Wilson / placebo), and the costed timer.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
