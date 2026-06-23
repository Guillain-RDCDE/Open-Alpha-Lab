"""Study 366 — Merger-Arbitrage (pocketing the deal spread).

After an all-cash takeover is announced, the target trades a few percent below the offer.
Buy it, hold to close, collect the spread — the classic "merger-arb coupon," sold as a
steady, market-neutral free lunch. The catch: when a deal *breaks*, the target snaps back
to its un-bid standalone level, a 20–40% loss in a day. Merger-arb is therefore short a
deal-break put — you sell insurance against deal failure, and the spread is the premium.

This study harvests realized arb returns from a hardcoded book of ~25 real announced
all-cash US M&A deals (a deliberate mix of clean closes and high-profile breaks), tests
whether the average deal pays *more* than fair compensation for the break tail, and uses a
deterministic synthetic deal book — with a known break probability and a planted edge knob
— as the positive control.

See :mod:`merger_arbitrage.data` (real deal book + price loader + synthetic control) and
:mod:`merger_arbitrage.strategy` (per-deal arb return, Welch t, bootstrap null, win-rate vs
break tail, costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
