"""Study 560 — Odd-Lot-Ratio: the classic 'dumb-money' contrarian fade, tested honestly.

The **odd-lot ratio** is one of Wall Street's oldest sentiment gauges. An *odd lot* is a trade
of fewer than 100 shares — historically the fingerprint of the small retail investor who could
not afford a round lot. The **odd-lot theory** (Drew 1950s; Garfield Drew) held that the odd-lot
crowd was *chronically wrong*: they bought at tops and sold at bottoms, so a spike in odd-lot
**buying** was a signal to *fade* (sell), and a spike in odd-lot **selling** a signal to buy.

Two structural shocks are supposed to have killed the signal:

1. **Decimalization (2001)** and the rise of algorithmic/HFT order-slicing — a single institutional
   parent order is now shredded into thousands of *odd-lot child orders*, so an odd lot is no longer
   'dumb retail money' at all. Most odd-lot volume today is machines, not mom-and-pop.
2. The odd-lot ratio was even **dropped from the consolidated tape** for years (odd lots were not
   reported to the SIP until the 2013–2014 Tick Size / odd-lot transparency changes), so the very
   series the theory rests on became unmeasurable and then unrepresentative.

Data availability: a clean, long, point-in-time **odd-lot ratio series is not available on a free,
no-key retail stack** (it needs proprietary TAQ/consolidated-tape odd-lot flags). So this study is
**synthetic-only** by construction, and — per the desk's house rule for synthetic-only studies
(lego-returns, whisky-cask, sneaker-resale) — it can never earn a `REAL` signal (that needs a
robust *t* ≥ 2 on a *real* tape). It is capped at `WEAK`/`NONE`, and the data-availability limit is
named openly on the SIGNAL axis.

What the engine *can* do, fully offline and deterministic, is answer the mechanical question the
folklore poses: **if** an odd-lot fade edge exists, does a clean contrarian test detect it, and does
it stay flat when the edge is absent (the modern, post-decimalization regime)? The single knob
``fade_alpha`` plants the effect: ``fade_alpha > 0`` is the old 'dumb-money' world where fading
odd-lot buying pays; ``fade_alpha = 0`` is the modern null.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
