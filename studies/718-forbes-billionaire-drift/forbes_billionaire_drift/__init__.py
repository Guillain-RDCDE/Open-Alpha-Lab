"""Study 718 — Forbes-Billionaire-Drift (buy the newly-minted billionaire's vehicle?).

The folklore: every spring Forbes publishes its World's Billionaires list, and a fresh
crop of *newly-minted* founders makes it for the first time. The trade sounds obvious —
buy the public **vehicle** behind each new name, ride the glow. We pin it down with a
textbook short-window event study around the list's publication date.

The catch the story hides is **reverse causality**: a founder joins the list *because*
their stock already multiplied — membership is a *consequence* of a run-up, not a cause
of the next one. So the pre-list window is huge and positive **by construction** (pure
selection / look-ahead you could never have traded), and the only honest question is the
**post-list drift**: is there abnormal return left over *after* the list is public, when
you could actually buy?

See :mod:`forbes_billionaire_drift.data` (hardcoded, cited table of newly-minted-founder
vehicles + annual list dates + yfinance loader + a deterministic synthetic control with a
plantable post-list drift) and :mod:`forbes_billionaire_drift.strategy` (market-model
abnormal returns, pre/announce/post CAR windows, Welch t / placebo null, one-day
execution lag, one-way costs).
"""

from . import data, strategy

__all__ = ["data", "strategy"]
