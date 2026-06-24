"""Study 427 — Rate of Change (ROC), the oldest momentum oscillator.

The folklore: ROC, the percentage price change over the last *N* days
(``100 * (P_t / P_{t-N} - 1)``), is the granddaddy of momentum indicators —
"go long when ROC is positive, step aside when it turns negative, and you ride
the trend while sidestepping the crashes." We take it literally as a daily
**long/flat** (and, where natural, **long/short**) timing rule on SPY, race its
**net** Sharpe against buy-and-hold *excess-vs-excess*, and ask the question the
folklore never does: does the oldest momentum tool actually beat the obvious
simpler benchmarks (a 200-day SMA cross, MACD, RSI) once you charge costs?

See :mod:`rate_of_change.data` (real SPY loader + deterministic synthetic tape
with a planted trend-persistence knob) and :mod:`rate_of_change.strategy`
(the ROC oscillator, the timing rules, HAC *t*, a sign-shuffle permutation
placebo, a cost sweep, and the simpler-benchmark race)."""

from . import data, strategy

__all__ = ["data", "strategy"]
