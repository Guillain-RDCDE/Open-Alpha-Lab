"""Study 790 — Composite Equity Issuance (Daniel-Titman 2006).

The 5-year *composite* equity issuance measure: the part of a firm's 5-year log
market-cap growth that is **not** explained by its own cumulative stock return —
i.e. net equity issuance (SEOs, buybacks, share-count drift) expressed in log
terms. High composite issuers are predicted to underperform low/negative issuers
(buyback-ers). We sort a ~40-name large-cap survivor basket on this 5y measure.

Two public sources, cached offline: SEC EDGAR ``CommonStockSharesOutstanding``
(``dei:EntityCommonStockSharesOutstanding`` fallback) for the point-in-time share
count, and yfinance raw + adjusted closes for the market cap and the return leg.
"""

from . import data, strategy  # noqa: F401
