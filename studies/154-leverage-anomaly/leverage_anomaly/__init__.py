"""Study 154 — Leverage-Anomaly: does financial leverage predict stock returns?

Penman, Richardson & Tuna (2007) decompose book-to-price into an operating and a
financial-leverage component and find that higher financial leverage is associated with
*lower* future returns — the opposite of what Modigliani-Miller risk-compensation
predicts.  We test the two simplest leverage proxies (LTD/Assets and Liabilities/Equity)
on the desk's shared EDGAR panel of S&P 500 companies.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
