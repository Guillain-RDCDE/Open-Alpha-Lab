"""Study 517 — Pre-FOMC-Drift.

Lucca & Moench (2015): a large slice of the equity premium is earned in the ~24 hours
*before* a scheduled FOMC announcement. We hardcode the FOMC announcement-date calendar,
tag the trading session immediately preceding each meeting, and ask whether SPY (and a
fixed survivor basket) earns an abnormal return on those pre-FOMC sessions versus every
other day — with a pre/post-2011 publication split (the McLean-Pontiff decay test) and an
overnight-vs-intraday decomposition (the study-specific third axis).

Distinct from study 67 (Fed-Drift, the original SPY single-day write-up) by its
**publication-decay + overnight-vs-intraday + cross-sectional** treatment, and from study
135 (FOMC-Cycle, the even-week *cycle* claim).
"""

from . import data, strategy

__all__ = ["data", "strategy"]
