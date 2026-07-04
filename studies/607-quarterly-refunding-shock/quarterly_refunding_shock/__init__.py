"""Study 607 — Quarterly Refunding Shock.

Does the Treasury's Quarterly Refunding Announcement move the long end? 106 hardcoded
QRA dates (2000 -> 2026, derived from the official TreasuryDirect auction records)
against ^TNX day-0 yield moves and TLT post-announcement holds, with the FOMC-overlap
decontamination the daily bar demands.
"""

from . import data, strategy  # noqa: F401
