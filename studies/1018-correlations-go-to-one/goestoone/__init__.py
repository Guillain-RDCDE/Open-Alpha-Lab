"""Study 1018 — Correlations Go to One.

"In a crisis all correlations go to one — diversification fails exactly when you need it."
It is one of the most repeated sentences in risk management, and it is half a measurement and
half a statistical illusion. Conditioning on a volatile period mechanically raises a measured
correlation even when the true correlation never moved (Boyer, Gibson & Loretan 1999; Forbes &
Rigobon 2002), so the naive "calm vs crisis" table overstates the effect by construction. This
study measures the naive number, subtracts exactly what the same procedure reports in a world
where correlation is constant by construction, and asks what is left — and whether what is left
actually hurts somebody holding a diversified book.

- :mod:`goestoone.data` — the real tapes (skfolio's pinned S&P 500 index and 20-stock survivor
  panel, via :mod:`quantlab.bundled`) and the deterministic synthetic generator with a planted
  crisis-correlation jump.
- :mod:`goestoone.strategy` — regimes, conditional correlations, the Forbes–Rigobon adjustment,
  the constant-correlation GARCH artefact benchmark, exceedance correlations, the diversified
  holder's arithmetic, the block bootstrap and the pre-registered verdict.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
