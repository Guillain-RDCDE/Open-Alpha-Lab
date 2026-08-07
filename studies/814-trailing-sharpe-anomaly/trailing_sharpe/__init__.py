"""Study 814 — Trailing-Sharpe Anomaly.

Risk-adjusted momentum: rank a liquid US cross-section on each name's **trailing 12-month
Sharpe ratio** (mean/std of daily returns, skipping the most recent month à la 12-1
momentum). Long the high-Sharpe names, short the low-Sharpe names — and ask honestly
whether risk-adjusting *adds* anything over plain 12-1 momentum, or is just momentum +
low-vol repackaged.

* ``data``     — the real cross-section (yfinance daily OHLC, cached under the study's
                 own ``_cache/`` through the ``quantlab.universe`` survivorship guard)
                 plus a deterministic seeded synthetic positive control (a planted
                 high-Sharpe->high-return relation, null at ``edge=0``).
* ``strategy`` — the trailing 12-1 Sharpe signal, the plain-momentum and low-vol
                 comparators, the point-in-time cross-sectional sort, the inference
                 primitives (Welch / one-sample / Newey-West HAC / Wilson / placebo), and
                 the costed timer.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
