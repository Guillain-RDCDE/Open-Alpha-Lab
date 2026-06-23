"""Study 381 — TIPS-Breakeven (does the bond market's inflation forecast time anything?).

The **breakeven inflation rate** — nominal Treasury yield minus the matched-maturity TIPS
yield — is the bond market's market-implied forecast of average inflation. Folklore says it
is a tradable macro timing signal: when breakeven is high/rising, buy inflation hedges (gold,
TIPS) and lean out of equities; when it is low/falling, do the opposite.

True breakeven (FRED ``T10YIE``) is not reliably fetchable here, so we CONSTRUCT a transparent
**proxy** from yfinance ETFs: the log ratio of an inflation-protected Treasury ETF (``TIP``) to
a duration-matched nominal Treasury ETF (``IEF``). When inflation expectations rise, TIP
outperforms IEF, so ``log(TIP/IEF)`` rises with breakeven. We test the proxy's **level**
(z-score) and **3-month change** against forward returns of TIPS, equities (SPY) and gold (GLD)
with HAC (Newey-West) predictive-regression *t*-stats, a placebo null, costs, and a synthetic
positive control.

The decisive finding: across 18 signal×asset×horizon tests the only ``|t| >= 2`` is a
**mechanical artifact** — the breakeven proxy predicting its *own* mean reversion (the signal is
built from TIP, so "high breakeven -> TIP reverts down" is autocorrelation, not a macro edge).
On genuinely independent targets (equities-over-bonds excess, gold) nothing robust survives,
and the one in-sample gold hit earns negative money as a strategy. Real-as-a-story,
weak-as-a-signal, mirage-as-a-trade.

See :mod:`tips_breakeven.data` (real ETF loader + breakeven proxy + deterministic synthetic
control) and :mod:`tips_breakeven.strategy` (signals, HAC predictive regression, placebo null,
win-rate, 1-day lag + costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
