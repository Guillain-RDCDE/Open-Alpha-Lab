"""Study 35 — Contango 🛢️: the commodity carry / roll-yield premium (long backwardated, short contangoed).

A commodity futures position earns a **roll yield** as it slides along the term-structure curve:
backwardated curves (front > deferred) roll up (positive carry), contangoed curves roll down (negative
carry). The documented premium (Gorton-Rouwenhorst 2006; Erb-Harvey 2006; Koijen et al. 2018) is that
backwardated commodities out-return contangoed ones, so a book long the most-backwardated and short the
most-contangoed harvests a real carry. The commodity sibling of Study 27 (Steamroller, FX carry) and a
cousin of Study 29 (Hedgers-Toll, commodity COT hedging pressure).

The cross-sectional bucket machinery is proved on an offline synthetic 12-commodity panel
(:mod:`contango.data`). The **real tape** is measured on the two liquid energy curves where the
term-structure roll is observable without a paid feed — front-month vs 12-month-laddered ETF pairs
(USO/USL, UNG/UNL) — in :mod:`contango.energy`; see ``docs/results.md``.
"""

from . import costs, data, energy, extension, strategy  # noqa: F401
