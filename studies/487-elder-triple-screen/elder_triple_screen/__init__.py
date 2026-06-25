"""Study 487 — Elder's Triple Screen trading system.

A mechanical, falsifiable encoding of Dr. Alexander Elder's three-screen method:

* **Screen 1 (the tide).** The higher-timeframe trend — here a *weekly* MACD-histogram
  slope. We only take longs when the weekly histogram is rising (the market tide is up).
* **Screen 2 (the wave).** A *daily* oscillator (Force Index / stochastic) that must be
  oversold *against* the up-tide — the pullback within the trend.
* **Screen 3 (the ripple).** A *breakout* trigger: the close clears the prior bar's high,
  confirming the pullback has turned. Enter at the next close (one documented lag).

The folklore: aligning three timeframes filters out noise and produces a high-odds long.
We test that as a forward-return study against a drift-matched random-entry baseline and a
screen-scramble placebo, with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
