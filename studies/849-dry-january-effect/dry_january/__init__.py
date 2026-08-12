"""Study 849 — Dry January / Veganuary.

Do the cultural "Dry January" (abstain from alcohol) and "Veganuary" (go plant-based)
waves show up as a January demand-shift in the stocks? A clean calendar-seasonality test
of the mean January (and February "hangover") **abnormal** returns of alcohol names
(``BUD, STZ, TAP, DEO, SAM``) vs a plant-based name (``BYND``) vs a consumer-staples
basket (``XLP``), all measured against the market (``SPY``), with a monthly-seasonality
*t*-stat (one-sample and Newey-West on a January dummy) and a placebo across the other
eleven calendar months.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
