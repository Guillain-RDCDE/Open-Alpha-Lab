"""Study 918 — Creation Halt.

When an ETF or ETN stops issuing new shares, the arbitrage that pins its price to the
value of what it holds is switched off in one direction. The folklore says the price
then floats up to a premium and collapses back when issuance resumes. This package
holds the hardcoded event list, the fund-versus-uncapped-twin spread, and the event
study that tests it.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
