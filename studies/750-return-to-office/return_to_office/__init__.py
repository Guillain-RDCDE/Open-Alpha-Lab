"""Study 750 — Return-to-Office (did RTO mandates move office REITs?).

Market lore, mid-2020s edition: every time a marquee employer orders workers **back to the
office** — Goldman calling WFH an "aberration", Musk's "40 hours or leave", Amazon's full
5-day return, the federal RTO order — the desks refill, office demand firms up, and beaten-
down **office landlords** (SL Green, Boston Properties, Vornado…) should get a reaction pop.

We make it falsifiable with a short-window **event study on a sector basket**. Over a
hardcoded, transparent table of ~26 real, dated RTO-mandate announcements, we measure the
office-REIT basket's **market-model abnormal return** (basket vs SPY) around each mandate,
split "full 5-day" (strict) from "hybrid" mandates, and judge both against a Welch *t*, a
placebo null sized to the event count, and a base-rate win-rate. The decisive finding is
statistical: ~two dozen events on a single sector that trades on **rates and secular WFH**
can't certify a fraction-of-a-percent reaction — and the worst-hit landlords (WeWork,
private CMBS-default towers) left the tape, so the *survivor* basket is a conservative read.

See :mod:`return_to_office.data` (the RTO calendar + office basket + benchmarks +
labelled Kastle occupancy proxy + deterministic synthetic control) and
:mod:`return_to_office.strategy` (basket market-model CAR, strict/hybrid buckets, placebo
null, costs).
"""

from . import data, strategy

__all__ = ["data", "strategy"]
