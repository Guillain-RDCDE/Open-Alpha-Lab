"""Study 441 — Camarilla Pivots.

The folklore: from yesterday's high/low/close you can compute a ladder of intraday
support/resistance levels (the **Camarilla** variant of the classic pivot family, using the
multipliers 1.1/12, 1.1/6, 1.1/4, 1.1/2). The strongest of them — **L3/H3** (the "reversal"
levels) and **L4/H4** (the "breakout" levels) — are sold as lines price *respects*: touch L3
and it bounces back toward the pivot; punch through H4 and it runs. Day-traders place their
whole book around these lines.

We test the single falsifiable core: **does price respect a Camarilla level more than it
respects a randomly-placed control line at the same distance?** Per session we compute the
prior-day Camarilla ladder, find the first intraday touch of L3 (long) / H3 (short), enter the
next bar, and measure the forward reversion toward the central pivot. The honest comparison is
against **random control levels** placed at the *same* distance from the open but at a shuffled
offset — the placebo that kills "a line is special" claims. Costs (intraday spread, paid both
ways) are charged loudly: intraday levels usually die to the spread.

See :mod:`camarilla_pivots.data` (real intraday loader via yfinance 5m/1m, cache-first, plus a
deterministic synthetic generator with a planted *respect-the-level* knob) and
:mod:`camarilla_pivots.strategy` (the Camarilla ladder, the touch/reversion event rule, the
random-control placebo, one-sample/HAC t, and costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
