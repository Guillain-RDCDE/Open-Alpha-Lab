"""Study 851 — Netflix Password Crackdown.

Netflix's 2023 paid-sharing ("password crackdown") was feared to spike churn but became
an **upside surprise** — a "scary policy that worked". This is a single-name
news-reaction **event study**: NFLX's abnormal returns (vs SPY / QQQ, a one-factor
market model) around the five public dates of the story — the Q1'22 first flag, the
2022-08 LatAm test, the 2023-05 broad US rollout, and the Q2/Q3'23 earnings that
confirmed the subscriber gains. Honest by construction: **N = 5 events → almost no
statistical power** — a case study, not a factor.

* ``data``     — the cached real tape (yfinance NFLX/SPY/QQQ under ``_cache/``), the
                 hardcoded public-record crackdown calendar, and a deterministic seeded
                 synthetic positive control (a planted event-day jump, null at edge=0).
* ``strategy`` — the market-model abnormal-return engine, the CAR paths, the cross-event
                 mean + one-sample t, a random-calendar placebo, the inference
                 primitives, and the costed long-only "buy the event" timer.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
