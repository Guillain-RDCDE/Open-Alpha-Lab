"""Study 363 — Post-Earnings-Announcement Drift (PEAD).

The folklore: after an earnings surprise a stock keeps **drifting the same way** for weeks —
good surprises keep rising, bad ones keep falling — so you can buy the beats and short the
misses and ride the drift. PEAD is one of the most-documented anomalies in finance (Ball &
Brown 1968; Bernard & Thomas 1989), but the textbook version is measured on the full
cross-section with quarter-deep histories.

We rebuild it cleanly on a **fixed large-cap basket**: per name we take the quarterly earnings
dates with the post-announcement **gap** as the surprise proxy, sort events into surprise
quintiles, and measure the forward 1/5/20/60-day **drift** of the top-minus-bottom long-short
— with a 1-day entry lag, a Welch / one-sample t, a label-shuffle placebo null, and one-way
costs × per-event turnover. The decisive question is not whether the drift exists (it does,
weakly) but whether it survives costs on a tradable large-cap universe.

See :mod:`pead_drift.data` (real basket+earnings loader + deterministic synthetic control with
a planted-drift knob) and :mod:`pead_drift.strategy` (gap surprise proxy, quintile long-short,
forward-drift inference, label-shuffle null, costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
