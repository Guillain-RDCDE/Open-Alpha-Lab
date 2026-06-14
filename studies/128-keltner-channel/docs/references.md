# References & literature map — Study 128 (Keltner-Channel)

## The claim under test

The Keltner Channel (Chester W. Keltner, *How to Make Money in Commodities*, 1960; later
popularized by Linda Bradford Raschke in the 1990s) is a volatility-based envelope around
an exponential moving average: mid = EMA(20), upper/lower = EMA(20) ± 2 × ATR(10).  Two
contradictory folk rules are widely claimed:

1. **Breakout rule (momentum framing):** *"When price closes above the upper Keltner Channel,
   the trend is strong — buy and ride it."*  Promoted in many retail trading guides and algo
   tutorials as a trend-confirmation filter.

2. **Reversion rule (mean-reversion framing):** *"When price closes below the lower Keltner
   Channel, it has stretched too far — buy the snap-back."*  The mirror-image claim, equally
   common, that the channel marks extremes that revert.

Both cannot be simultaneously valid on the same instrument at the same time horizon.  Our study
tests them honestly against a random-entry control that removes the instrument's unconditional
drift, and finds neither arm adds exploitable alpha after this correction.

## Why the channel looks promising — the real effects it leans on

- **Trend persistence / time-series momentum.** Moskowitz, Ooi & Pedersen (2012), *Time Series
  Momentum* (Journal of Financial Economics), document positive autocorrelation in commodity
  and equity returns at the monthly horizon — a theoretical basis for the breakout arm.  At
  the daily-horizon our synthetic positive control confirms the Keltner breakout only helps
  when bar-level momentum is actually present.

- **Short-term mean reversion.** De Bondt & Thaler (1985), *Does the Stock Market Overreact?*
  (Journal of Finance), document longer-horizon reversion; at shorter horizons Jegadeesh (1990),
  *Evidence of Predictable Behavior of Security Returns* (Journal of Finance), shows weekly
  reversals — a basis for the reversion arm.  Again, our positive control confirms the
  reversion arm only outperforms when mean-reversion is planted in the tape.

- **Volatility channels in general.** Chester W. Keltner (1960), *How to Make Money in
  Commodities* — the original construction used a 10-day simple MA of the high-low midpoint
  and a 10-day simple MA of the high-low range (not ATR).  The modern form (EMA + ATR,
  popularized by Raschke and then by various system-trading books) is a refinement, making
  the channel adaptive to recent volatility.

## The trap this study is really about

- **Positive gross mean ≠ signal.** Both arms show positive raw gross means and even positive
  HAC *t*-stats — but these merely reflect the long-run equity drift of the instruments in the
  basket.  A random-entry control matched in trade count earns a statistically equivalent mean.
  Novy-Marx & Velikov (2016), *A Taxonomy of Anomalies and Their Trading Costs* (Review of
  Financial Studies), discuss how failure to benchmark against the unconditional drift inflates
  apparent performance of signals with positive skew to the underlying.

- **The contradiction trap.** When the same channel band is claimed to signal *both* momentum
  (upper pierce) and mean reversion (lower pierce), at least one claim is logically constrained
  to be wrong.  This study is a controlled demonstration of that contradiction on real data.
  The related [Study 104 — Bollinger-Reversion](../../104-bollinger-reversion/) makes the same
  point for Bollinger Bands.

- **ATR-normalized channels and the survivorship horizon problem.** The ATR-based width makes
  the channel widen during volatile periods (e.g. 2008, 2020), which means channel-pierces
  are mechanically clustered around crisis regimes.  Returns after crisis-era lower-band
  pierces may look strong because of the large unconditional recovery; this does not imply
  the channel is doing anything beyond marking a volatile regime.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.summarize`](../keltner_channel/strategy.py) implements the local NW estimator.

- **Average True Range.** Wilder (1978), *New Concepts in Technical Trading Systems* — the ATR
  is the risk unit for the channel width in [`strategy.atr`](../keltner_channel/strategy.py).

- **Exponential moving average.** Standard `pandas.ewm(span=20, adjust=False)` — the EMA for
  the channel mid band in [`strategy.ema`](../keltner_channel/strategy.py).

- **Random-entry control.** The discipline of pinning a signal arm against a size-matched
  random-entry baseline (so the instrument's drift is priced in) is the desk's standard
  for daily-frequency studies.  See also [Study 104 — Bollinger-Reversion](../../104-bollinger-reversion/)
  and [Study 72 — Loaded-Dice](../../72-loaded-dice/) for the same framework.

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), five liquid tickers: SPY, QQQ, IWM, GLD, EEM.
  Window 2005-01-03 to 2026-06-12 (~21 years, ~5,395 bars per ticker).  Content fingerprints
  in [`docs/results.md`](results.md).  The offline reproducible core and the test-suite run on
  the deterministic [`data.synthetic_daily`](../keltner_channel/data.py) generator, never the
  network.

## Related desk studies

- **[Study 104 — Bollinger-Reversion](../../104-bollinger-reversion/)**: the Bollinger Band
  version of the same reversion/breakout contradiction — same logical structure, same verdict.
- **[Study 72 — Loaded-Dice](../../72-loaded-dice/)**: the SMA(5/10) intraday crossover —
  the "does the filter add anything over a coin?" framework this study uses.
- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: the daily 50/200 golden cross — the
  moving-average-crossover family, one timeframe up from the intraday version.
- **[Study 78 — Crossed-Wires](../../78-crossed-wires/)**: MACD crossover daily — the
  momentum/trend-filter family with a more complex signal construction.
