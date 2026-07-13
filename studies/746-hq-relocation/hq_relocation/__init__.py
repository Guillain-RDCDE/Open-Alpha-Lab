"""Study 746 — HQ-Relocation (do headquarters-move announcements signal anything?).

Market lore: a company that announces relocating its **headquarters** — an inversion
abroad (Medtronic -> Ireland), or a jump to a lower-tax U.S. state (Tesla, Oracle,
Caterpillar, Chevron -> Texas) — is sending a signal. One camp says *buy it*: the market
prices in the tax saving and the cost cut. The other says *fade it*: a splashy new address
is management theatre that masks a weak business. Which is it — signal, or distraction?

We make it falsifiable with a textbook short-window **event study** over a hardcoded,
transparent table of ~20 documented HQ relocations, 2010-2025 (tagged tax/incentive vs
other rationale). Around each announcement the **cumulative abnormal return** (CAR) is the
stock's return minus a **market-model** fit (``stock = alpha + beta*SPY``) on a clean
pre-event window; we test the announcement CAR, the tax-minus-other gap, and a longer
post-announcement **drift** leg entered the day after. The decisive finding is statistical:
a couple-dozen events per bucket is too few to certify anything, and the holdable drift is
zero — on the tape, an HQ move is a **non-event**, neither the signal nor the distraction
the folklore needs.

See :mod:`hq_relocation.data` (the relocation table + real loader + deterministic synthetic
control) and :mod:`hq_relocation.strategy` (market-model CAR, placebo null, drift, costs).
"""

from . import data, strategy

__all__ = ["data", "strategy"]
