"""Study 721 — Most-Admired (does making Fortune's list predict returns?).

Market lore, two ways. The optimist's version: Fortune's *World's Most Admired Companies*
are the best-run firms on earth — great management, great products, great moats — so owning
the list is owning quality, and quality compounds (the **admiration premium**). The
contrarian's version (Antunovich–Laster–Mishra 2000 vs Statman–Fisher–Anginer 2008): a firm
becomes *admired* only after a long run of good news, so the label marks a stock that is
already loved, richly priced, and due to **mean-revert** — buy the *spurned*, not the
admired.

We make both falsifiable with a small **characteristic sort**: a hardcoded, cited table of
the list's perennial All-Stars, an equal-weight admired book benchmarked against the market
(SPY), a market-model **alpha**, and a **Newey–West (HAC)** *t* on the monthly excess
return. The decisive honesty problem is named up front and on the Signal axis: a *current*
most-admired list is **look-ahead selection** — Apple/Microsoft/Nvidia are on it *because*
they already won — so any raw out-performance is survivorship + factor beta before it is an
"admiration" effect. A publication-lagged variant (own a name only *after* Fortune first
crowns it) is the honest test, and a placebo of random large-cap books sizes the luck.

See :mod:`most_admired.data` (the admired/spurned tables + real loader + deterministic
synthetic control) and :mod:`most_admired.strategy` (portfolio excess returns, market-model
alpha, HAC inference, placebo null, costs).
"""

from . import data, strategy

__all__ = ["data", "strategy"]
