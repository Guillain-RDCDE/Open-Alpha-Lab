"""Study 516 — Dividend-Month-Premium (Hartzmark & Solomon 2013).

Stocks earn abnormally high returns in the *calendar months they are PREDICTED to pay a
dividend* — a price-pressure story driven by yield-seeking demand that clusters around the
predictable payment schedule. We rebuild the test on a fixed large-cap survivor basket:
from each name's dividend history we learn which calendar months it pays, flag every future
month a name is *predicted* to pay (using only past payments — no look-ahead), and compare
the in-month return against the same name's non-dividend months.

Engine:
  * ``data``     — yfinance prices + per-name dividend payment dates, cached to ``_cache/``;
                   plus a deterministic synthetic panel with a *planted* dividend-month premium.
  * ``strategy`` — predicted-month flags (past-only), the per-event premium + one-sample t,
                   a label-shuffle placebo, a tradable monthly overlay with costs, and the
                   synthetic faithful-engine / power control.
"""
