"""Study 35 — Contango 🛢️: the commodity carry / roll-yield premium (long backwardated, short contangoed).

A commodity futures position earns a **roll yield** as it slides along the term-structure curve:
backwardated curves (front > deferred) roll up (positive carry), contangoed curves roll down (negative
carry). The documented premium (Gorton-Rouwenhorst 2006; Erb-Harvey 2006; Koijen et al. 2018) is that
backwardated commodities out-return contangoed ones, so a book long the most-backwardated and short the
most-contangoed harvests a real carry. The commodity sibling of Study 27 (Steamroller, FX carry) and a
cousin of Study 29 (Hedgers-Toll, commodity COT hedging pressure).

Real roll yield needs the **term structure** (front + deferred contracts) — not available in this
sandbox, which caches only front-month continuous returns. So, following Study 27's honesty pattern, the
offline synthetic control proves the machinery and the real-tape run is **pending a term-structure fetch**
(see ``docs/results.md``).
"""

from . import costs, data, extension, strategy  # noqa: F401
