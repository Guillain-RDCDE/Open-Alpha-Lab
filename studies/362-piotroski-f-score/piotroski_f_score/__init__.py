"""Study 362 — Piotroski F-Score (the 9-point fundamental-health checklist).

Joseph Piotroski's F-Score (2000) is a 9-point accounting checklist — profitability,
leverage/liquidity, and operating efficiency — said to "separate winners from losers."
We rebuild all nine binary points from EDGAR companyfacts for a fixed basket of large/mid
caps, sort firms by score, and test a high-minus-low (8-9 vs 0-2) cross-sectional
portfolio held the following calendar year, with a Newey-West t, a placebo null of random
same-size portfolios, a monotonicity check across the 0..9 ladder, an execution lag, and
costs with borrow on the short leg.

See :mod:`piotroski_f_score.data` (EDGAR/yfinance real tape + deterministic synthetic
control) and :mod:`piotroski_f_score.strategy` (F-score construction, long-short, HAC t,
placebo, costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
