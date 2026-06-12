"""Study 71 — Ambush: four dead-net S&P 500 edges, traded only at their confluence.

The bench's inversion: every short-horizon edge it certified gross died under daily
turnover (studies 01, 03, 13, 19, 42). Instead of averaging the signals (study 38's
mistake), the ambush book stays flat until k of them fire on the same close — rarity,
not signal strength, is the cost defence — under study 16's vol targeting and a hard
1%-of-NAV daily risk budget. Pre-registered in ``docs/preregistration.md``.
"""

from . import data, signals, strategy, synth  # noqa: F401
