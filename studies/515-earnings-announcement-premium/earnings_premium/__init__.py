"""Study 515 — Earnings-Announcement Premium (Frazzini-Lamont / Beaver).

A disproportionate share of a stock's total return is earned in the few days *around* its
scheduled earnings announcement. We compare each name's mean daily return inside a narrow
**announcement window** against its mean daily return on **non-announcement** days, and turn
the difference into a tradable long-the-announcers calendar strategy.

Distinct from PEAD (study 363): PEAD is the *drift after* a surprise sorted on the sign of the
surprise; this is the *level premium on the announcement days themselves*, irrespective of
surprise sign.
"""

from . import data, strategy  # noqa: F401
