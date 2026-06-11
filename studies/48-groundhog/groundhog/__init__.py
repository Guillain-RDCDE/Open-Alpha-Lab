"""Study 48 — Groundhog: return seasonality, the anomaly that actually replicates.

A stock tends to repeat its calendar-month performance: a name that has historically done well in
Marches keeps doing well in Marches (Heston & Sadka 2008). It sounds like numerology — and yet on a
large-cap survivor cross-section it is real, significant (t≈4), specific to the *same* month (a clean
control fails), and undecayed. The honest question isn't whether it exists, but whether its monthly
turnover lets you keep it.
"""

from . import data, strategy  # noqa: F401
