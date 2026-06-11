"""Study 48 — Groundhog: return seasonality, the anomaly that actually replicates.

A stock tends to repeat its calendar-month performance: a name that has historically done well in
Marches keeps doing well in Marches (Heston & Sadka 2008). It sounds like numerology — and yet it is
significant (t≈4), specific to the *same* month (a control that shares the panel's bias fails), and
undecayed. Measured on a large-cap survivor cross-section, so the magnitude is an upper bound — the
existence is what the control and Heston-Sadka's CRSP evidence support. The honest question isn't
whether it exists, but whether one-way monthly turnover (~3.2x NAV) and the short book let you keep it.
"""

from . import data, strategy  # noqa: F401
