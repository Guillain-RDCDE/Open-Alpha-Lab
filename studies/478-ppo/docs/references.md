# References & literature map — Study 478 (Percentage Price Oscillator)

## The claim under test

- **The folklore.** Compute the PPO = 100·(EMA12 − EMA26)/EMA26 and a 9-period EMA of it (the
  *signal line*). A **bullish crossover** — the PPO line rising above its signal line — is a buy;
  the bearish crossover is a sell. The selling point of the PPO over MACD is that it is a
  *percentage*, so its readings are **comparable across instruments and across time** (a $600 SPY
  and a $30 GLD give comparable PPO values, while their raw MACDs are on incomparable scales).
  This is a retail/technician staple built into TradingView, StockCharts, MetaTrader and every
  charting suite.
- **The source.** The **MACD** was created by **Gerald Appel** in the late 1970s; the
  **Percentage Price Oscillator** is its normalized variant, popularized through StockCharts'
  ChartSchool and Martin Pring's and John Murphy's technical-analysis texts. Appel's *Technical
  Analysis: Power Tools for Active Investors* (2005) is the primary lineage for the crossover
  rule; Thomas Aspray's signal-line work is the origin of the histogram/crossover idiom. The PPO
  itself is a straightforward rescaling: same EMAs, divided by the slow EMA.
- **The thesis question.** Because the PPO *is* MACD divided by a slow-moving EMA, the natural
  test is whether the **normalization adds edge** — does dividing by EMA26 change *when* the
  crossover fires in a way that improves timing, or only the *units*? We run the raw MACD
  crossover side-by-side as the comparator.

## Why this is a mechanical-proxy study

The PPO crossover is already fully objective (no eyeballing) — the only design choices are the
standard 12/26/9 parameters, which we fix to the textbook defaults a proponent would accept:

- **Objective oscillator.** Standard recursive EMAs (`adjust=False`); the PPO and its signal are
  read on the close of *t*; a crossover needs only *t* and *t-1* — no look-ahead.
- **The honest baseline.** The only meaningful comparison on an upward-drifting index is the
  **random-entry** control (same instrument, epoch and hold), because *any* long-only entry rule
  inherits the drift. We add a **shuffled-sign placebo** that destroys the specific crossover
  structure while keeping the marginal of |PPO − signal| — the direct test of "does the crossover
  structure matter?" — and a **raw-MACD comparator** for the normalization question.

## Why the high one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample *t*
  of a long-only entry rule against **zero** measures that drift, not the rule. See Fama & French
  on the equity premium; the desk's standing rule is *signal-vs-baseline*, never *signal-vs-zero*.
- **Data snooping on chart tools.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis*, JF) formalize testing technical rules against a properly matched null; Sullivan,
  Timmermann & White (1999, *Data-Snooping, Technical Trading Rule Performance, and the
  Bootstrap*, JF) and White (2000, *A Reality Check for Data Snooping*, Econometrica) show how
  trend-fitted rules manufacture significance unless raced against a fair benchmark.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the crossover-vs-random difference.

## Method lineage (the desk's shared engine)

- **PPO / MACD construction.** [`strategy.ppo_lines`](../ppo/strategy.py),
  [`strategy.macd_lines`](../ppo/strategy.py) — the oscillator and signal line.
- **Crossover entries + forward returns + HAC t + random baseline.**
  [`strategy.ppo_cross_entries`](../ppo/strategy.py),
  [`strategy.forward_returns`](../ppo/strategy.py), [`strategy.hac_t`](../ppo/strategy.py),
  [`strategy.run_experiment`](../ppo/strategy.py).
- **Crossover placebo.** [`strategy.shuffled_sign_placebo`](../ppo/strategy.py) — permute the
  sign of PPO − signal, keep its marginal.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../ppo/data.py) plants a real
  post-crossover continuation (knob `edge`); with `edge = 0` the detector must NOT manufacture
  significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../../103-macd`](../../103-macd) (and the wider MACD/crossover family) — the un-normalized
  parent of this study; the PPO is the same crossover, rescaled.
- [`../../450-andrews-pitchfork`](../../450-andrews-pitchfork) — the same random-entry-baseline +
  geometry-placebo idiom on a different chart tool; the template for this study's construction.
- The broader **technical-indicator zoo** (CCI, Aroon, TRIX, Vortex, …) — most land
  None × Mirage for the same reason: an oscillator fitted to past price re-describes the trend.
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting) frame why a
  signal-vs-zero *t* is not enough; the PPO crossover is a clean live example of beta masquerading
  as a momentum signal — and a clean test that normalization changes units, not edge.
