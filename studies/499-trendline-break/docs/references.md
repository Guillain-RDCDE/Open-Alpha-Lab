# References & literature map — Study 499 (Trendline-Break)

## The claim under test

- **The folklore.** In an uptrend, connect the recent **swing lows** with a rising
  **trendline** (support). While price holds above the line the trend is "intact"; a
  **confirmed close below the line** signals that *support has broken* — exit longs, and (the
  bolder version) go short, because "a broken trendline forecasts a turn down." This is the
  single most-taught construct in technical analysis, present in every charting suite
  (TradingView, MetaTrader, StockCharts) and every introductory TA text.
- **The source.** The trendline is the founding object of the **Dow Theory** lineage and of
  Edwards & Magee's *Technical Analysis of Stock Trends* (1948), the canonical text that
  codified support/resistance lines and "the break of the trendline" as a reversal signal.
  John Murphy's *Technical Analysis of the Financial Markets* (1999) restates it as a core
  rule; Schabacker's 1930s chart-pattern work is the earlier lineage. The "two/three lows
  define the line, the break confirms the change of trend" formulation is textbook Edwards &
  Magee.
- **Variants.** Hand-drawn vs least-squares trendlines, "valid break" filters (a close beyond
  the line by X%, or N consecutive closes, or a volume-confirmed break), channel lines, and
  the symmetric resistance-line break. All are **affine variants of the same fit-a-line-through-
  pivots geometry** and inherit the same drift confound tested here.

## Why this is a "theory" / mechanical-proxy study

The trendline is *semi-subjective*: a discretionary chartist chooses which lows to connect and
when a break "counts". Following the desk's design for this kind, we encode the **tightest
mechanical rule a proponent would accept** and state the irreducible subjectivity explicitly:

- **Objective pivots.** Confirmed **fractal** swing lows (a local minimum with *k*
  strictly-higher bars on each side), only usable *k* bars later — a documented confirmation
  lag, no look-ahead.
- **Objective line.** Ordinary least-squares fit through the *n* most-recent confirmed lows,
  required to be **rising** (an uptrend support line); no hand-picking which lows to connect.
- **The honest baseline.** The only meaningful comparison on a drifting index is the
  **random-entry** control (same instrument, epoch and hold). We add a **shuffled-slope
  placebo** that re-fits the line from permuted swing-low prices, destroying the geometry
  (slope/level) while keeping the price marginal — the direct test of "does the trendline
  matter?"

Hand-drawn trendlines add *hindsight* (a free parameter — you keep redrawing until the line
"works"), which can only inflate in-sample fit; the mechanical version is the charitable
**upper bound** on the method.

## Why the high one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample
  *t* of a directional entry rule against **zero** measures that drift, not the rule. The break
  rule is *bearish* (it shorts the index), so on an up-drifting tape it fights the tide — but
  the honest test is still break-vs-random, never break-vs-zero. See Fama & French on the
  equity premium; the desk's standing rule is *signal-vs-baseline*.
- **Data snooping on chart tools.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis: Computational Algorithms, Statistical Inference, and Empirical Implementation*, JF)
  formalize testing chart patterns against a properly matched null; Sullivan, Timmermann &
  White (1999, *Data-Snooping, Technical Trading Rule Performance, and the Bootstrap*, JF) and
  White (2000, *A Reality Check for Data Snooping*, Econometrica) show how line-fitted rules
  manufacture significance unless raced against a fair benchmark.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the break-vs-random difference.

## Method lineage (the desk's shared engine)

- **Confirmed-fractal swing lows + rolling OLS trendline.**
  [`strategy.find_swing_lows`](../trendline_break/strategy.py),
  [`strategy.build_trendlines`](../trendline_break/strategy.py) — the mechanical geometry with
  the confirmation lag baked in.
- **Forward-return + HAC t + random baseline.**
  [`strategy.forward_returns`](../trendline_break/strategy.py),
  [`strategy.hac_t`](../trendline_break/strategy.py),
  [`strategy.run_experiment`](../trendline_break/strategy.py).
- **Geometry placebo.** [`strategy.shuffled_slope_placebo`](../trendline_break/strategy.py) —
  permute swing-low prices, keep positions and marginals.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../trendline_break/data.py)
  plants a real post-break downward continuation (knob `edge`); with `edge = 0` the detector
  must NOT manufacture significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced
  by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../450-andrews-pitchfork`](../450-andrews-pitchfork) — the sibling median-line channel
  tested with the identical random-entry + geometry-placebo idiom.
- [`../104-bollinger-reversion`](../104-bollinger-reversion) — the same "the band/channel
  reverts price" folklore tested with the random-entry baseline.
- The broad **technical-indicator zoo** (Supertrend, Donchian, Keltner, the chart-pattern
  studies) — most land None × Mirage for the same reason: an object fitted to past price
  re-describes the trend.
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting) frame why a
  signal-vs-zero *t* is not enough; the trendline break is a clean live example of a fitted
  line carrying no forecasting information.
