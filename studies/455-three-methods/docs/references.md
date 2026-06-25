# References & literature map — Study 455 (Rising/Falling Three Methods)

## The claim under test

- **The folklore.** The *rising three methods* is a long white candle, then three small candles
  that drift down but stay **inside** the first candle's high–low range, then a long white candle
  closing **above** the first's close. The *falling three methods* is the bearish mirror. The
  three small candles are read as a brief **pause** (profit-taking, not a reversal), and the
  fifth candle's break past the first **confirms the trend continues** — so you trade in the
  trend's direction. It is a staple "continuation" pattern in every candlestick scanner.
- **The source.** These are classic Japanese *san-poh* ("three methods") patterns. **Steve
  Nison** introduced them to Western readers in *Japanese Candlestick Charting Techniques*
  (1991) and *Beyond Candlesticks* (1994); they trace to the rice-trading lore attributed to
  **Munehisa Homma** (18th c.). Greg Morris's *Candlestick Charting Explained* and the
  candlestick recognisers in **TA-Lib** (`CDLRISEFALL3METHODS`) and TradingView encode the same
  rule used here.
- **Variants / cousins.** The **mat-hold** is a near-identical continuation pattern with a
  smaller pullback; **separating lines** and **upside/downside-gap three methods** are affine
  cousins. All share the same "pause then continuation" thesis and inherit the same drift
  confound tested here.

## Why this is a "theory" / mechanical-proxy study

The three-methods is *semi-subjective*: a discretionary trader judges what counts as a "long"
candle, a "small" candle, and "inside the range." Following the desk's design, we encode the
**tightest mechanical rule a proponent would accept** and state the irreducible knobs explicitly:

- **Objective candles.** Anchor body > 1.0 × trailing-20 average body ("long"); middle bodies
  < 0.7 × anchor body ("small"); middles contained in the anchor range with a **10% wick
  tolerance** (exact wick containment is rare on daily bars — this is the only charitable knob,
  set to 0 for strict containment).
- **No look-ahead.** The five-candle pattern is fully *closed* at bar *t*; the signal is read on
  the close of *t* and the trade entered at the close of *t+1*. No pivot-confirmation lag is
  even needed — the pattern is self-contained.
- **The honest baseline.** On a directional (long-rising / short-falling) rule the only
  meaningful comparison is a **random-entry** control with the *same long/short mix*, so the
  drift is netted out. We add a **shuffled-date placebo** that destroys the five-candle geometry
  while keeping the price marginal — the direct test of "does the shape matter?"

## Why a high one-sample t (or a high win-rate) is not evidence

- **Drift / beta via the long/short mix.** US equity indices have a positive unconditional daily
  mean; a directional rule inherits it through its net long exposure. A one-sample *t* against
  **zero** measures that, not the pattern. The desk's standing rule is *signal-vs-baseline*,
  never *signal-vs-zero* — here the baseline is matched in count *and* long/short mix.
- **Data snooping on chart patterns.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis*, Journal of Finance) formalize testing chart patterns against a properly matched
  null; Sullivan, Timmermann & White (1999, *Data-Snooping, Technical Trading Rule Performance,
  and the Bootstrap*, JF) and White (2000, *A Reality Check for Data Snooping*, Econometrica)
  show how pattern rules manufacture significance unless raced against a fair benchmark. Marshall,
  Young & Rose (2006, *Candlestick technical trading strategies: Can they create value for
  investors?*, Journal of Banking & Finance) find no value in candlestick signals once properly
  tested — directly on point.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the pattern-vs-random difference.

## Method lineage (the desk's shared engine)

- **Mechanical pattern detection.** [`strategy.three_methods_signals`](../three_methods/strategy.py),
  [`strategy.pattern_entries`](../three_methods/strategy.py) — the five-candle geometry, signed.
- **Forward-return + HAC t + matched random baseline.**
  [`strategy.forward_returns`](../three_methods/strategy.py),
  [`strategy.hac_t`](../three_methods/strategy.py),
  [`strategy.run_experiment`](../three_methods/strategy.py).
- **Geometry placebo.** [`strategy.shuffled_body_placebo`](../three_methods/strategy.py) —
  same count + long/short mix on random dates, geometry destroyed.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../three_methods/data.py)
  plants a real post-pause continuation (knob `edge`); with `edge = 0` the detector must NOT
  manufacture significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced
  by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../450-andrews-pitchfork`](../450-andrews-pitchfork) — the same random-entry + geometry-placebo
  idiom on a drawing tool; also None × Mirage.
- The broader candlestick / chart-pattern zoo (engulfing, doji, head-and-shoulders, the
  402–450 technical-analysis lot) — most land None × Mirage for the same reason: a pattern
  fitted to past price re-describes the trend rather than forecasting it.
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting, multiple-testing)
  frame why a signal-vs-zero *t* or a high win-rate is not enough; the three-methods is a clean
  live example of a famous pattern with no forecasting content.
