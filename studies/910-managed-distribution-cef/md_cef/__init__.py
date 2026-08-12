"""Study 910 — Managed-Distribution CEF.

Do persistent-discount, big-managed-distribution closed-end funds hand a buyer a real
excess-of-cash, risk-adjusted return vs the asset class — or does NAV erosion / embedded
leverage cost eat the payout, leaving a levered-beta clone with a yield sticker?

``data``     — real tape (yfinance total-return closes) + a seeded synthetic control.
``strategy`` — basket construction, excess-vs-excess Sharpe race, HAC t's, beta decomposition,
               bootstrap Sharpe CI, drawdown / calendar year, era cut, and the costed net.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
