"""Study 807 — Salience-Theory Returns.

Cosemans & Frehen (2021), applying Bordalo-Gennaioli-Shleifer salience: over the trailing
month a name whose **salient days were UP relative to the market** (a high salience-theory
value ST) is bid up by salience-thinking investors and goes on to earn **lower** returns. We
sort a liquid US cross-section on its trailing ST and measure the forward return of a
long-low-ST / short-high-ST book.

* ``data``     — the real cross-section (yfinance daily OHLC, cached under the study's own
                 ``_cache/`` through the ``quantlab.universe`` survivorship guard) plus a
                 deterministic seeded synthetic positive control (a planted negative
                 salient-upside->return relation, null at ``edge=0``).
* ``strategy`` — the salience-theory value signal, the point-in-time cross-sectional sort,
                 the inference primitives (Welch / one-sample / Newey-West HAC / Wilson /
                 placebo), and the costed timer.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
