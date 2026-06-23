"""Study 392 — Glassdoor-Sentiment (do the happiest employees beat the market?).

Edmans (2011) found that Fortune's "100 Best Companies to Work For" earned a real,
risk-adjusted long-short premium. The retail folklore — "buy the firms with the best
Glassdoor ratings" — is its descendant. Real employer-review data is not free, so this study
sorts a fixed large-cap basket by a **constructed, clearly-labelled employee-satisfaction
proxy** (NOT scraped Glassdoor data) and measures the happiest-minus-grumpiest long-short
spread, with one-period execution lag, costs, a t-stat, a bootstrap Sharpe CI, and a
sign-flip placebo null.

The decisive finding is methodological: a proxy that is independent of returns by
construction yields a long-short premium indistinguishable from zero — a proxy/synthetic-only
result can never back a REAL stamp. The synthetic positive control (with a planted
happiness->return edge) proves the engine recovers a true edge when one exists.

See :mod:`glassdoor_sentiment.data` (real price loader + constructed sentiment + synthetic
control) and :mod:`glassdoor_sentiment.strategy` (quintile sort, long-short, inference,
placebo null, costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
