"""Study 725 — "Eggflation" (trade the avian-flu egg-price spike via Cal-Maine).

Egg prices go vertical every time bird flu culls the flock (2015, 2022–23, 2024–25) —
so surely you can *front-run* it in the pure-play egg stock, **Cal-Maine Foods (CALM)**,
the largest US shell-egg producer. We test the strongest tradable version: does a
**labelled, cited, approximate USDA/BLS retail egg-price series** (a proxy for the tape,
not a live feed) *predict* CALM's forward return, and does an egg-momentum timer beat
buy-and-hold net of costs?

See :mod:`eggflation.data` (hardcoded, cited egg-price series + yfinance CALM/SPY +
a deterministic synthetic positive control) and :mod:`eggflation.strategy`
(contemporaneous vs predictive HAC regressions, a circular-shift placebo, the
reverse-lead test — does the *stock* lead the government print? — and the timer)."""

from . import data, strategy

__all__ = ["data", "strategy"]
