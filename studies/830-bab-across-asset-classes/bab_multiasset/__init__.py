"""Study 830 — BAB Across Asset Classes.

Frazzini-Pedersen "betting against beta", lifted from single stocks to the
**multi-asset** level: estimate each asset's beta to an equal-weight multi-asset
market portfolio, then go **long the low-beta assets (levered to unit beta)** and
**short the high-beta assets (de-levered to unit beta)** so the book is ex-ante
beta-neutral. If the security-market line is too flat *across asset classes*, this
BAB factor earns a positive risk-adjusted spread.

Public surface:

* ``data``     — the yfinance cross-asset tape + the seeded synthetic control.
* ``strategy`` — the rolling betas, the BAB construction, the inference primitives,
  the costed timer, and the synthetic detector.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
