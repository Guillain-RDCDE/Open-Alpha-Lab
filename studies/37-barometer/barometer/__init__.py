"""Study 37 — Barometer: fundamental macro-momentum and inflation hedging across assets.

The trend in fundamental macro data (growth, inflation) is a slow, cross-asset predictor: go long the
assets favoured by improving macro momentum, and tilt toward *real* assets when inflation is rising
(Brooks & Moskowitz 2017, "Macro Momentum"; Neville et al. 2021, "The Best Strategies for Inflationary
Times"; Kakushadze-Serur §19.2/§19.3). The macro state needs FRED series that are unreliable to fetch in
this sandbox, so — exactly like Study 27 (Steamroller) — the offline core is a **synthetic control** that
proves the machinery, the verdict is grounded in that control plus the literature, and the real-tape run
is **pending a reliable FRED macro fetch** (see ``docs/results.md``).
"""

from . import costs, data, extension, strategy  # noqa: F401
