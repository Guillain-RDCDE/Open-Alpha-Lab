# References — Study 431 (Schaff Trend Cycle)

## The claim's source

- **Doug Schaff (2008)** — the originator. The Schaff Trend Cycle was introduced by Doug Schaff,
  a foreign-exchange trader, as a refinement of the MACD: run a stochastic over the MACD line and
  double-smooth it so the oscillator "cycles" between 0 and 100 and **turns earlier** than the
  MACD. The marketing line, repeated across charting platforms, is that it is a *"faster MACD"*.
- **Investopedia — "Schaff Trend Cycle (STC)"** — the canonical retail write-up: the 23/50/10
  default settings, the 0.5 smoothing factor, and the 25/75 buy/sell thresholds we implement
  verbatim. <https://www.investopedia.com/articles/forex/10/schaff-trend-cycle-indicator.asp>
- **TradingView / StockCharts indicator docs** — the widely-mirrored implementation reference
  (double stochastic of MACD with EMA smoothing) that our `schaff_trend_cycle()` reproduces.

## The benchmark it claims to beat

- **Gerald Appel (1979)** — the MACD (Moving Average Convergence/Divergence), the indicator STC
  is built from and explicitly markets itself against. Our head-to-head uses the classic 12/26/9
  crossover.

## Why trend-timing rules struggle on equity indices

- **Brock, Lakonishok & LeBaron (1992)**, *Simple Technical Trading Rules and the Stochastic
  Properties of Stock Returns*, *Journal of Finance* 47(5) — the seminal test of moving-average
  rules; later shown to be heavily exposed to data-snooping.
- **Sullivan, Timmermann & White (1999)**, *Data-Snooping, Technical Trading Rule Performance,
  and the Bootstrap*, *Journal of Finance* 54(5) — applies White's Reality Check to the universe
  of technical rules and finds the apparent out-performance largely vanishes after correcting for
  selection. The methodological backbone for treating any single charting rule sceptically.
- **Park & Irwin (2007)**, *What Do We Know About the Profitability of Technical Analysis?*,
  *Journal of Economic Surveys* 21(4) — a survey: profits, where they exist, are concentrated in
  pre-1990 data, FX and futures, and erode under realistic costs on liquid equity indices.

## Shared-method citations (the desk's inference engine)

- **Newey & West (1987)**, *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix*, *Econometrica* 55(3) — the HAC standard errors
  behind every *t*-stat here (a step-position return series is strongly autocorrelated).
- **White (2000)**, *A Reality Check for Data Snooping*, *Econometrica* 68(5) — the rationale for
  the permutation placebo: a single rule pulled from a large family must clear a selection-aware
  null, not just *t* > 2.
- **Politis & Romano (1994)**, *The Stationary Bootstrap*, *JASA* 89(428) — the block-resampling
  idea our position-block permutation borrows to preserve holding-period structure.

## Related desk studies

- [`../../106-supertrend/`](../../106-supertrend/) — another lagging ATR-band trend rule on
  equities; same null verdict.
- [`../../178-cci/`](../../178-cci/) — the Commodity Channel Index overbought/oversold rule; also
  no edge over a coin on equity tapes.
- [`../../104-bollinger-reversion/`](../../104-bollinger-reversion/) — the band-touch reversion
  rule whose "edge" is shown to be bull-market drift, not the indicator.
- [`../../425-detrended-price-oscillator/`](../../425-detrended-price-oscillator/) — a sibling
  oscillator teardown in the same family.
