"""Study 747 — Founder-Led-Premium (do founder-run firms beat professional-CEO peers?).

The claim is old and well-cited: Fahlenbrach (2009) reports founder-CEO firms earn
positive abnormal returns, and Bain's "founder's mentality" thesis turned it into
management folklore. We test it as a clean **long/short characteristic sort**: a
transparent, hardcoded **founder-led** basket (long) vs a matched **professional-CEO**
basket (short), equal-weighted, rebalanced monthly, and ask whether the spread's
market-model **abnormal return** (Jensen alpha, Newey-West HAC t) is real — or whether it
is just **survivorship** (the founder firms we remember in 2024 are the ones that lived)
plus **tech-sector beta concentrated in one or two names**.

See :mod:`founder_led_premium.data` (hardcoded founder / professional baskets + yfinance
monthly loader + a deterministic synthetic control with a plantable founder alpha) and
:mod:`founder_led_premium.strategy` (equal-weight baskets, CAPM alpha with a Newey-West
HAC t, leave-one-out jackknife, a random-label placebo null, costs + borrow)."""

from . import data, strategy

__all__ = ["data", "strategy"]
