"""Study 375 — VXX-Roll-Decay (shorting volatility ETPs for the contango carry).

VIX-future ETPs (VXX, VIXY) hold a constant-maturity short-dated VIX-futures position
and roll it daily. When the VIX term structure is in **contango** (the usual state, here
~92% of days), each roll sells a cheaper near future and buys a richer far future, so the
ETP **bleeds** value over time — VIXY's split-adjusted price fell from ~6.3e5 to ~22 over
2011-2026. Shorting that bleed is the famous "short-vol carry": a steady drip of roll
yield, paid for by a fat **left tail** when volatility spikes (Volmageddon 2018, COVID
2020, the 2024 yen-carry unwind).

The believers' framing: *"is shorting VXX a steady carry, or picking up nickels in front
of a steamroller?"* We answer both halves separately, the desk way:

* **Signal** — is the roll-decay carry statistically real? (HAC/Newey-West t on the real
  tape; a placebo-shuffle null; a synthetic positive control with a planted carry knob.)
* **Tradability** — does it survive costs (borrow on the short, rebalancing turnover) and,
  above all, the **crash tail** (skew, kurtosis, drawdown, single-day wipeouts)?

See :mod:`vxx_roll_decay.data` (real VIXY/VIX-term loader + a deterministic synthetic
contango-carry-with-a-crash generator) and :mod:`vxx_roll_decay.strategy` (the short-carry
book, HAC inference, placebo null, costs with borrow, and the tail diagnostics)."""

from . import data, strategy

__all__ = ["data", "strategy"]
