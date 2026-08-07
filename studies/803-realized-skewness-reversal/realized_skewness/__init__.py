"""Study 803 — Realized-Skewness Reversal.

Amaya, Christoffersen, Jacobs & Vasquez (2015): stocks with high recent **realized
return skewness** go on to earn **lower** returns — a lottery-like right tail is
over-priced. We sort a liquid US cross-section on its trailing realized skewness and
measure the forward return of a long-low-skew / short-high-skew book.

* ``data``     — the real cross-section (yfinance daily OHLC, cached under the study's
                 own ``_cache/`` through the ``quantlab.universe`` survivorship guard)
                 plus a deterministic seeded synthetic positive control (a planted
                 negative skew->return relation, null at ``edge=0``).
* ``strategy`` — the trailing realized-skewness signal, the point-in-time
                 cross-sectional sort, the inference primitives (Welch / one-sample /
                 Newey-West HAC / Wilson / placebo), and the costed timer.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
