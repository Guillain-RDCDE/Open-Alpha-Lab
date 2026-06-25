# References & literature map — Study 489 (Chaikin Oscillator)

## The claim under test

- **The folklore.** The **Chaikin Oscillator** is the momentum of the Accumulation/Distribution
  Line: `EMA3(ADL) − EMA10(ADL)`. The lore — taught by its creator and repeated on every charting
  site — is that **A/D momentum leads price**: a cross **above zero** means the short-term
  accumulation EMA has overtaken the long-term one, buyers are gathering, and a price advance is
  imminent. So a cross above zero is a **buy** (and a cross below zero a sell). It is built into
  TradingView, MetaTrader, StockCharts and every charting suite.
- **The source.** **Marc Chaikin** created the oscillator in the 1970s–80s, building on Larry
  Williams' Accumulation/Distribution concept and refining the *Money Flow Multiplier*
  `((C−L)−(H−C))/(H−L)` that weights each bar's volume by where price closed in its range. The
  modern popular write-ups (Investopedia, StockCharts' ChartSchool, and the volume-indicator
  chapters of John Murphy's *Technical Analysis of the Financial Markets*) restate the rule.
- **Lineage.** The A/D line descends from Joseph Granville's On-Balance-Volume idea (cumulating
  signed volume) and Williams' Accumulation/Distribution; the Chaikin Money Flow (CMF) and the
  Chaikin Oscillator are the two momentum/oscillator forms of the same ADL. All share the premise
  that *volume reveals accumulation before price moves* — the premise tested here.

## Why this is a mechanical-proxy study

The cross-above-zero rule is fully objective once the EMA spans (3, 10) are fixed at their
textbook defaults, so there is no discretionary anchoring to encode — but two design choices must
be stated:

- **Causality.** Every EMA uses `adjust=False` (past-only); the cross is read on the close of *t*
  and the trade is entered at *t+1*'s close. No future bar touches a signal.
- **Volume proxy.** The Chaikin oscillator needs volume, which the shared desk price-cache does
  not store. We attach a **deterministic, look-ahead-free** proxy that is a monotone function of
  the bar's range (`high−low`) using only current-bar data. It cannot fabricate a forward lead;
  it reproduces the *shape* of the indicator (closes near the high → positive money flow) faithfully
  enough to test the rule, and the placebo + random baseline neutralize it regardless.

## Why the high one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample *t*
  of a long-only entry rule against **zero** measures that drift, not the rule. See Fama & French
  on the equity premium; the desk's standing rule is *signal-vs-baseline*, never *signal-vs-zero*.
- **Data snooping on chart/volume tools.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis*, Journal of Finance) formalize testing technical signals against a properly matched
  null; Sullivan, Timmermann & White (1999, *Data-Snooping, Technical Trading Rule Performance,
  and the Bootstrap*, JF) and White (2000, *A Reality Check for Data Snooping*, Econometrica) show
  how rules fitted to past price/volume manufacture significance unless raced against a fair
  benchmark.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the cross-vs-random difference.

## Method lineage (the desk's shared engine)

- **Indicator + cross rule.** [`strategy.chaikin_oscillator`](../chaikin_oscillator/strategy.py),
  [`strategy.cross_above_zero_entries`](../chaikin_oscillator/strategy.py) — the causal EMA(3)−EMA(10)
  of the ADL with the next-close entry lag.
- **Forward-return + HAC t + random baseline.** [`strategy.forward_returns`](../chaikin_oscillator/strategy.py),
  [`strategy.hac_t`](../chaikin_oscillator/strategy.py), [`strategy.run_experiment`](../chaikin_oscillator/strategy.py).
- **Geometry placebo.** [`strategy.scrambled_mfm_placebo`](../chaikin_oscillator/strategy.py) —
  permute the per-bar Money Flow Multipliers, keep volume and the marginal.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../chaikin_oscillator/data.py)
  plants a real accumulation-leads-price effect (knob `edge`); with `edge = 0` the detector must
  NOT manufacture significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. Volume is the deterministic range proxy. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../../186-obv`](../../186-obv) — On-Balance-Volume, the ancestor of the A/D line, tested with
  the same "volume leads price" premise and random-entry idiom.
- [`../../450-andrews-pitchfork`](../../450-andrews-pitchfork) — the template for this study's
  random-entry baseline + geometry-placebo design.
- The broader technical-indicator zoo (CCI, Aroon, TRIX, Vortex, …) — most land None × Mirage for
  the same reason: an indicator fitted to past price/volume re-describes the trend rather than
  forecasting it.
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting) frame why a
  signal-vs-zero *t* is not enough; the Chaikin oscillator is a clean live example of beta
  masquerading as a leading volume indicator.
