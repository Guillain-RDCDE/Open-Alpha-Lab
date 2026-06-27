"""Study 530 -- Book-To-Market-Value (the canonical Fama-French HML value factor).

Public surface:

- ``data``     -- synthetic + real (yfinance) book-to-market panels, study-local cache.
- ``strategy`` -- the B/M quintile sort, long-value / short-growth hedge, HAC inference,
                  a label-shuffle placebo null, costs x turnover (+ borrow), and a
                  deterministic synthetic positive control.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
