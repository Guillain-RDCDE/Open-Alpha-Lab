"""Study 789 — SUE Earnings-Surprise Drift (the classic PEAD, sorted on SUE).

Does the post-earnings-announcement drift line up with the **standardized unexpected
earnings** (SUE = the seasonal-random-walk EPS surprise `EPS_q − EPS_{q−4}` scaled by the
rolling volatility of the last ~8 such surprises)? We build a clean event study on a fixed
large-cap basket: per name we pull every quarterly diluted-EPS figure from EDGAR
(frame-tagged calendar quarters, with the 10-Q/10-K filing date), form SUE, sort filings
into SUE terciles, and measure the forward 1 / 2 / 3-month drift of a top-minus-bottom
long-short — entering the close **one day after the filing is public** (no look-ahead).

Deliberately distinct from its siblings: 363-pead-drift (the price-gap CAR around the
announcement, no fundamentals), 369-earnings-revision-momentum (analyst revisions),
534-revenue-surprise-drift (SUR on *revenue*, not EPS). This is the **SUE-sorted** portfolio
drift — the Bernard-Thomas (1989) original.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
