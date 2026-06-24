"""Study 413 — Bull Flag (the classic bullish-continuation figure).

The folklore: after a sharp rally (the **flagpole**), a stock pauses in a brief, shallow
consolidation that drifts gently down or sideways (the **flag**) — and when it closes back above
the flag's high (the **breakout**), it *resumes the prior uptrend* for a second leg. So you buy
the breakout and ride it. The bull flag is one of the most-taught continuation figures (Edwards &
Magee 1948; Bulkowski's *Encyclopedia of Chart Patterns*).

We rebuild it cleanly with the closest **mechanical** definition we can write down — a steep
flagpole over a short window, a tight flat-to-down flag that gives back only a fraction of the
pole, and a confirmed close above the flag's high — across **SPY + 29 US large-caps** on daily
OHLC. For each confirmed breakout we measure the forward 5/10/20/40-day return as an **excess
over each name's own base rate** (does the figure beat buy-and-hold for that name?), with a
one-sample / HAC *t*, a **same-tape label-shuffle placebo** that corrects for the names' own
up-drift, a **down-break symmetry** myth-check, and a deterministic synthetic positive control.
Chart figures are partly subjective — we test one transparent definition and say so.

See :mod:`bull_flag.data` (real basket loader + deterministic synthetic control with a planted
post-breakout edge knob) and :mod:`bull_flag.strategy` (the flag detector, the breakout timing
rule, forward-return inference, the same-tape placebo, and costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
