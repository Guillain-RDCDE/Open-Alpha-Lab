"""Study 755 — JOLTS-Quits (does a falling quits rate warn before equities weaken?).

The labour-nowcasting folklore: the JOLTS **quits rate** is worker confidence made
visible — people quit voluntarily only when they're sure of something better — so when
quits turn **down**, confidence is fading and the equity cycle (especially cyclicals) is
supposedly about to soften. We rebuild the believers' signal on the monthly quits-rate
tape (a hardcoded snapshot of FRED ``JTSQUR``, Dec 2000..May 2026) and measure forward
1/3/6/12-month SPY (and cyclical-minus-defensive) returns conditional on falling vs
rising quits momentum, against the unconditional base rate, with a Welch t, a placebo
null, a lead/lag scan, and a tradable timing overlay — under the honest **2-month JOLTS
release lag** (the print for month t is only public in month t+2).

The decisive findings are about *timing, publication delay and tradability*, not
direction: quits and the cycle are tangled, but the quits rate is a slow, coincident-to-
lagging tell published ~6 weeks late — not an early-warning you can allocate to.

See :mod:`jolts_quits.data` (hardcoded quits snapshot + SPY/XLY/XLP loaders +
deterministic synthetic control) and :mod:`jolts_quits.strategy` (momentum signal,
release-aware forward-return inference, placebo null, lead/lag, costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
