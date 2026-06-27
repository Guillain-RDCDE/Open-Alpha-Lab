"""Study 531 -- Enterprise-Multiple (EV/EBITDA value sort).

The enterprise multiple, EV/EBITDA, is Loughran & Wellman's (2011) value signal:
buy firms with a low multiple (cheap relative to operating earnings), short firms
with a high multiple (expensive). The denominator EBITDA and the capital-structure
adjustment (EV = market cap + total debt - cash) distinguish it from a simple P/E or
book-to-market sort.

Two layers:
- ``data``     -- the real tape (yfinance fundamentals, cached to this study's _cache/)
                  and a deterministic synthetic panel with a tunable, dial-able premium.
- ``strategy`` -- the monthly cross-sectional sort, the long-short hedge, one-sample /
                  HAC t-stats, a label-shuffle placebo null, costs x turnover (+ borrow),
                  and the synthetic positive control.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
