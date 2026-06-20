"""Study 339 — Convertible-Bonds: do convertibles give equity upside with bond downside?

A convertible bond is a corporate bond with an embedded equity call. The marketing line
is *convexity* — you ride the stock up but a "bond floor" cushions the way down. We test
that claim on the **CWB** ETF (SPDR Bloomberg Convertible Securities) against a plain
**stock/bond blend** built to match its equity exposure, and ask whether the curvature
(positive gamma vs the index) is real or just a re-priced linear mix.

Distinct from Study 97 (Balancing-Act, fixed 60/40 vs 100% stocks): there the object is
*diversification*; here the object is **convexity** — an asymmetric, non-linear payoff.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
