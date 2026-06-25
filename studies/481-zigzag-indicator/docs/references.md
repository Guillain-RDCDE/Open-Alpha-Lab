# References & literature map — Study 481 (ZigZag Indicator)

## The claim under test

- **The folklore.** The ZigZag draws straight legs between alternating swing highs and lows,
  starting a new leg only when price **reverses by more than `x%`** off the prior extreme,
  filtering out the noise in between. "The ZigZag identifies turns" — when the indicator turns
  **up** off a swing low, that low was a tradable bottom, so go long. This is a retail/technician
  staple built into MetaTrader (the standard `ZigZag` indicator), TradingView, Thinkorswim and
  every charting suite, and the scaffolding under Elliott-wave and Gartley/harmonic-pattern tools.
- **The source.** The ZigZag is an old charting primitive with no single named inventor; it was
  popularised through the **Merrill / Arthur Sklarew** "swing filter" lineage and standardised in
  trading platforms (the MetaTrader/MetaQuotes `ZigZag` with parameters *Depth, Deviation,
  Backstep*; Welles Wilder-era swing charts share the idea). Its modern role is as the pivot
  engine for **Elliott Wave** (R. N. Elliott via Frost & Prechter, *Elliott Wave Principle*) and
  for **harmonic patterns** (H. M. Gartley, *Profits in the Stock Market*, 1935; Scott Carney,
  *Harmonic Trading*). The common write-ups (Investopedia, StockCharts ChartSchool, the
  MetaTrader docs) all restate the `x%`-reversal rule.
- **The repaint problem.** Every honest description of the ZigZag warns that **the last leg
  repaints**: the indicator anchors the newest leg to the latest extreme, and erases/redraws it
  as price moves, so a swing low is only *final* once price has rebounded `x%`. A backtest that
  reads the finished ZigZag and "buys the last low" peeks at the future. This study trades only
  **confirmed** legs (the bar where the `x%` rebound completes), which is the only look-ahead-free
  encoding a proponent should accept.

## Why this is a "theory" / mechanical-proxy study

The ZigZag is *fully* mechanical given its threshold, so there is no eyeballing to remove — the
only subjectivity is the threshold `pct` and the repaint discipline. Following the desk's design:

- **Objective swings.** Threshold ZigZag with `pct = 5%` (the classic default); consecutive
  same-direction moves are absorbed into the running extreme, exactly as the platform indicator.
- **No repaint.** A swing low is *confirmed* at the first bar whose close is `pct` above it — a
  documented confirmation lag, never the future-peeking final pivot.
- **The honest baseline.** The only meaningful comparison on an upward-drifting index is the
  **random-entry** control (same instrument, epoch and hold), because *any* dip-confirmed-buy
  inherits the drift. We add a **relabelled-leg placebo** that keeps the confirmation *timing*
  and the price marginal but randomises which confirmations are called "lows" — the direct test
  of "does the up/down geometry matter, or is any confirmation date as good?"

## Why the high one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample *t*
  of a long-only entry rule against **zero** measures that drift, not the rule. See Fama & French
  on the equity premium; the desk's standing rule is *excess-vs-excess* and *signal-vs-baseline*,
  never *signal-vs-zero*.
- **Data snooping on chart tools.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis: Computational Algorithms, Statistical Inference, and Empirical Implementation*, JF)
  formalize testing chart patterns against a properly matched null; Sullivan, Timmermann & White
  (1999, *Data-Snooping, Technical Trading Rule Performance, and the Bootstrap*, JF) and White
  (2000, *A Reality Check for Data Snooping*, Econometrica) show how trend-fitted rules manufacture
  significance unless raced against a fair benchmark.
- **Repaint as look-ahead.** The classic ZigZag's apparent precision is an artefact of plotting
  the *finished* line; the confirmation lag here is the antidote, and the contrast between the
  repainting `zigzag_line` (illustration only) and the confirmed-leg entry rule is the point.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the up-leg-vs-random difference.

## Method lineage (the desk's shared engine)

- **Threshold ZigZag with confirmation.** [`strategy.zigzag_confirmations`](../zigzag_indicator/strategy.py),
  [`strategy.confirmed_uppleg_entries`](../zigzag_indicator/strategy.py) — the mechanical swing
  filter with the repaint lag baked in; [`strategy.zigzag_line`](../zigzag_indicator/strategy.py)
  draws the (unsafe, repainting) display line for illustration only.
- **Forward-return + HAC t + random baseline.** [`strategy.forward_returns`](../zigzag_indicator/strategy.py),
  [`strategy.hac_t`](../zigzag_indicator/strategy.py), [`strategy.run_experiment`](../zigzag_indicator/strategy.py).
- **Geometry placebo.** [`strategy.shuffled_leg_placebo`](../zigzag_indicator/strategy.py) —
  keep confirmation timing + marginals, randomise the up/down labels.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../zigzag_indicator/data.py)
  plants a real post-turn drift (knob `edge`); with `edge = 0` the detector must NOT manufacture
  significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced
  by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../../450-andrews-pitchfork`](../../450-andrews-pitchfork) — a sibling chart-geometry tool
  tested with the identical random-entry + geometry-placebo idiom; also None × Mirage.
- [`../../104-bollinger-reversion`](../../104-bollinger-reversion) — the "the band/channel reverts
  price" folklore tested with the random-entry baseline.
- The broader technical-indicator zoo (CCI, Aroon, Supertrend, Renko, point-and-figure…) — most
  land None × Mirage because an indicator fitted to past price re-describes the trend.
- The **research-method demos** (look-ahead, curve-fitting, data-mining-roulette) frame why a
  signal-vs-zero *t* is not enough; the ZigZag's repaint is a textbook live look-ahead trap.
