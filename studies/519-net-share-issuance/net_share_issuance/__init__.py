"""Study 519 — Net-Share-Issuance (Pontiff-Woodgate / Daniel-Titman composite issuance).

The factor: firms that **issue** shares (dilute) underperform; firms that **buy back**
(shrink the share count) outperform. We compute each firm's *split-adjusted* net change in
shares outstanding over the prior year and sort the cross-section: long the low-issuance
(buyback) names, short the high-issuance (diluting) names.

Two tapes:

* ``net_share_issuance.data`` — the real Yahoo tape (split-adjusted shares outstanding via
  ``Ticker.get_shares_full`` + split history, plus daily adjusted closes), cache-first under
  this study's own ``_cache/``.
* ``net_share_issuance.strategy`` — the annual cross-sectional sort, the long-short book, the
  one-sample t, a label-shuffle placebo null, costs × turnover + short borrow, robustness
  cuts, and a deterministic synthetic positive control.

Distinct from [Study 368 — Buyback-Drift](../368-buyback-drift/), which times discrete buyback
*authorization announcements* as an event study. Here we measure the **realised** net change
in the share count — the full issuance axis, dilution included — as a cross-sectional factor.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
