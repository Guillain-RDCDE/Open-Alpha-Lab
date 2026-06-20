"""Study 319 — Lockup-Expiry: do newly-IPOd stocks sag when the 180-day lockup unlocks?

An *event study* around the IPO lock-up expiry date (canonically 180 calendar days, here
modelled as ~125 trading days post-listing). Distinct from Study 219 (IPO-Pop), which
measures the first-day pop and the multi-year long-run drift; this study isolates the
*supply shock* of insider shares becoming sellable at the lock-up expiry and asks whether
there is an abnormal price sag in a tight window around that single calendar-known date.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
