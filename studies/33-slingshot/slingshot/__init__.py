"""Study 33 — Slingshot: short-horizon cross-sectional mean-reversion within a single group (the S&P 500).

The equity counterpart to Study 32 (Rip-Tide). Where Rip-Tide fades each *futures* market against its own
past and finds nothing, Slingshot fades each *stock* against the **cross-section** of its peers — buy the
relative laggards, short the relative leaders, dollar-neutral. Short-term reversal is a famously real
anomaly in single stocks (Jegadeesh 1990, Lehmann 1990); the question is whether the gross edge survives
the brutal turnover the signal demands.
"""

from . import costs, data, extension, strategy  # noqa: F401
