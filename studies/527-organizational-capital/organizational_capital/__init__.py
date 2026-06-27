"""Study 527 -- Organizational-Capital (Eisfeldt-Papanikolaou 2013).

The org-capital stock is built by perpetual inventory of SG&A spending, scaled by
total assets, and sorted cross-sectionally. High org-capital firms are predicted to
earn higher returns -- they bear more risk tied to a key, mobile input (talent).

- ``data``     -- EDGAR fundamentals (SG&A + Assets) and yfinance prices, cached to _cache/.
- ``strategy`` -- org-capital perpetual inventory, the cross-sectional long-short sort,
                  one-sample HAC t, label-shuffle placebo, costs x turnover + borrow, and a
                  deterministic synthetic positive control.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
