"""Study 389 — Name-Change-Effect (do theme-chasing rebrands pop, then give it back?).

Market lore: a company that renames toward whatever's hot — append ``.com`` in 1999, add
``Blockchain`` in 2017, slap ``AI`` on it in 2023 — gets a sentiment **pop** on the rename
and then **gives it all back** as the hype cools. Long Island Iced Tea -> Long Blockchain
(+200% in a day) is the poster child.

We make it falsifiable with a small **event study** over a hardcoded, transparent table of
~25 real theme-chasing rebrands across the three waves. Around each rename we measure the
abnormal (excess-of-SPY) return on a short **pop** leg and a longer **fade** leg, then judge
both against a Welch *t*, a placebo null sized to the event count, and a base-rate win-rate.
The decisive finding is statistical: ~25 events is too few to certify either leg, and the
surviving tape (the worst give-backs delisted, leaving no series) is biased *against* the
fade — so the romantic "pop then dump" is real-as-anecdote, unproven-as-edge, and untradable
once small-cap costs hit four crossings per event.

See :mod:`name_change_effect.data` (the rebrand table + real loader + deterministic
synthetic control) and :mod:`name_change_effect.strategy` (event windows, abnormal-return
inference, placebo null, costs).
"""

from . import data, strategy

__all__ = ["data", "strategy"]
