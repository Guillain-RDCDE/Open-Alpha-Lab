"""Social-Oracle — does following a viral stock guru actually pay?

A reproducible, honest stress-test of the "buy what the influencer mentions" idea —
the desk's first study whose *trigger lives in the information flow*, not in the
price/vol series (the twin, in attention space, of Studies 02–03). The viral case
is a cashtag feed (@aleabitoreddit-style mentions distilled into `$SYMBOL`s by the
open-source "Serenity skill" repos), but the engine is generic: hand it any
``(timestamp, ticker)`` feed and it measures, around each mention, the only things
that matter — does the name beat a *random day* in the same universe, does it add
anything over the *momentum it already had*, and does the apparent pop survive the
**fade**, the micro-cap **costs**, and the **selection** baked into "we only hear
about the calls that worked".

The unit of analysis is the **abnormal return** (the name minus its market), in the
textbook event-study convention (additive CAR), so the pop-then-fade signature is
read off directly.

Public surface (import from the sub-modules):
    from social_oracle import data, mentions, eventstudy
    from social_oracle import benchmark, backtest, robustness

This package is a research and teaching tool. It tests a *phenomenon* (social
trading), not a person, and it is NOT investment advice.
"""

__version__ = "0.1.0"

__all__ = [
    "data",
    "mentions",
    "eventstudy",
    "benchmark",
    "backtest",
    "robustness",
]
