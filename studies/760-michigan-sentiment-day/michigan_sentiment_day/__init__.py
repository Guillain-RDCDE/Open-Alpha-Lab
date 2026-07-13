"""Study 760 — Michigan-Sentiment-Day (a release-day drift *and* a bottom-timer test).

Two believer claims about the University of Michigan Consumer Sentiment release, each
tested on its own tape:

* **Release-day drift.** The preliminary sentiment print (mid-month Friday, 10:00 ET) is
  a market-moving number, so SPY should react on release day and drift with the surprise.
* **Low-then-rising marks bottoms.** The contrarian folklore (Fisher-Statman): buy stocks
  when sentiment is low and turning up.

We rebuild both on real tapes — a hardcoded, public monthly snapshot of FRED ``UMCSENT``
(FRED's CSV endpoint is firewalled here, mirroring Study 385's hardcoded ``IC4WSA``) and
cached yfinance SPY (daily for the release-day event study, month-end for the regime
test). The headline result: the release-day drift is a non-event, and the bottom-timer
*looks* strong on a naive t but is a handful of clustered post-crash recoveries that a
12-month circular block bootstrap can't certify — real-as-lore, weak-as-edge, and a
mirage to trade.

See :mod:`michigan_sentiment_day.data` (hardcoded sentiment snapshot + release-date proxy
+ SPY loaders + deterministic synthetic control) and
:mod:`michigan_sentiment_day.strategy` (release-day drift, regime split, block bootstrap,
episode count, timing overlay)."""

from . import data, strategy

__all__ = ["data", "strategy"]
