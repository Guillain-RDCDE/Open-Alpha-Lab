"""Study 800 — High-Frequency (Weekly) Cross-Sectional Reversal.

Sort a liquid US cross-section on last week's 5-day return; long the losers, short the
winners; measure next week's loser-minus-winner spread, and ask the only question that
matters for a *weekly* reversal: **does it survive a bid-ask-bounce haircut?**

This is the weekly cousin of study 329 (one-month reversal). The engine and honest
controls live in :mod:`hf_reversal.strategy`; the tapes (a shared daily S&P 500 panel
resampled to weekly, plus a deterministic seeded synthetic panel with tunable *reversal*
and *bid-ask-bounce* knobs) live in :mod:`hf_reversal.data`.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
