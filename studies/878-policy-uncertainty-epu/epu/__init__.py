"""Study 878 — Economic Policy Uncertainty (EPU).

Baker, Bloom & Davis' newspaper-based **Economic Policy Uncertainty** index is the
canonical "how uncertain is policy right now" gauge. The famous claim has two legs:

1. **The vol story** — high EPU should predict **higher forward equity volatility**.
2. **The risk-premium story** — high EPU should predict **higher forward returns** as
   compensation for bearing that uncertainty.

The desk's honest prior is that uncertainty indices are mostly **contemporaneous**: they
spike *with* drawdowns and vol, not ahead of returns. We test both legs with predictive
regressions (forward SPY return AND forward realized vol on the uncertainty level/change),
Newey-West HAC *t*, an era cut, a permutation placebo, and a seeded synthetic control.

* ``data``     — the real tape (SPY + ^VIX via yfinance, cached under ``_cache/``) and the
                 EPU series. ``fetch_epu`` tries the real Baker-Bloom-Davis feed
                 (policyuncertainty.com / FRED ``USEPUINDXM``); when that endpoint is
                 unreachable it falls back to a **documented market-based proxy** built from
                 real VIX, clearly labelled (never the newspaper index). Plus a seeded
                 synthetic positive control (null at ``edge=0``).
* ``strategy`` — the monthly frame builder, the forward-return / forward-realized-vol
                 predictive regressions with HAC *t*, the inference primitives
                 (one_sample_t / welch_t / newey_west_t / predictive_reg / wilson_interval /
                 placebo / timer_stats / synthetic_detect), and the costed timer.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
