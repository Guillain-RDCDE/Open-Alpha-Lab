"""Study 251 — Crypto-Reversal.

A teardown of the central *reproducible* claim of the QuantNest-7 paper
(Babayev & Aliyev, 2026, SSRN 6818558): that **short-term reversal — not
momentum — prices the cross-section of crypto returns** (their REV factor,
*t* = 6.19, +151%/yr). The paper's two genuinely novel factors (on-chain
*quality* and *value*) rely on paid Coin Metrics data and undisclosed
proprietary composites, so they are not reproducible here; the reversal,
momentum and volatility factors are price-only and *are* reproducible on free
daily closes. This study rebuilds the price-based slice on a public yfinance
panel and asks the honest question the paper's own limitations section raises:
how much of the spread survives survivorship bias and the brutal turnover cost
of a daily-rebalanced long-short crypto book?
"""

from . import data, strategy  # noqa: F401
