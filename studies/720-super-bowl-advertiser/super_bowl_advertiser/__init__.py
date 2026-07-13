"""Study 720 — Super-Bowl-Advertiser (do companies that buy a Super Bowl ad drift up?).

Market lore (and a real academic paper): a company that runs a **Super Bowl commercial**
buys a burst of national attention, and its stock **drifts up in the days after the game** —
the "big-ad signal". Fehle, Tsyplakov & Zdorovtsov (2005) found significantly positive
abnormal returns around Super Bowl ads; the folklore turns that into a tradable calendar,
one you'd rerun every February.

We make it falsifiable with a small **event study** over a hardcoded, transparent table of
~32 real *listed* Super Bowl advertisers (advertiser-year events, 2015-2024). Around each
game we measure the abnormal (excess-of-SPY) return on a short **drift** leg — the days
right after the game, the "big-ad signal" — and a longer **hold** leg, with a one-day entry
lag (you can only act at Monday's close, the Sunday-night ad already public). We then judge
both against a Welch *t*, a placebo null sized to the event count, a base-rate win-rate, and
the (large) cost of a 30-name ad calendar.

The decisive findings are statistical: ~32 events is far too few to certify a few-tenths-of-a-
percent drift, and the surviving tape is biased *for* the story — the loudest advertisers that
went to zero (Pets.com, Computer.com, Kozmo.com — the dot-com Super Bowl class of 2000)
**delisted** and leave no series, so a survivor-only drift near zero is a conservative
refutation. Real-as-a-2005-paper, absent-as-an-edge, untradable once the ad-buyers' costs and
the missing corpses are counted.

See :mod:`super_bowl_advertiser.data` (the advertiser table + real loader + deterministic
synthetic control) and :mod:`super_bowl_advertiser.strategy` (event windows, abnormal-return
inference, placebo null, costs).
"""

from . import data, strategy

__all__ = ["data", "strategy"]
