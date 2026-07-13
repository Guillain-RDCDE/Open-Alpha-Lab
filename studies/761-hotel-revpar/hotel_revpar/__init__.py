"""Study 761 — Hotel-RevPAR 🏨 (does RevPAR momentum *lead* hotel REITs?).

The pitch: hotel **RevPAR** (Revenue Per Available Room) — the lodging industry's
canonical demand gauge, published monthly by STR / CoStar — is a leading read on the
travel cycle. When RevPAR momentum turns up, the story goes, be long hotel REITs, because
accelerating travel demand isn't fully priced yet.

STR's monthly RevPAR is proprietary, so we hardcode a small, **clearly-labelled
approximate reconstruction** anchored to STR/CoStar-reported annual U.S. RevPAR (with the
2020 COVID collapse and 2021 recovery set to the reported national monthly path). We align
it to **HST** (Host Hotels & Resorts, the flagship lodging REIT) and to an equal-weight
lodging-REIT basket, with a strict STR release lag, and ask whether RevPAR YoY momentum
**leads** the hotel tape or merely **echoes** what the equity already discounted.

The decisive object is direction of causation: hotel equities are forward-looking claims
on future room revenue, so the stock tends to *lead* the reported RevPAR, not the reverse
— a coincident-or-lagging gauge dressed as a leading indicator.

See :mod:`hotel_revpar.data` (the hardcoded RevPAR proxy, cached HST + basket loaders, and
a deterministic synthetic positive control with a PLANTED lead) and
:mod:`hotel_revpar.strategy` (YoY momentum, forward-return inference, Welch t / HAC
predictive regression / placebo null, lead-lag cross-correlation, costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
