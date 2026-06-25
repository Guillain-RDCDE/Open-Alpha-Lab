# References & literature map — Study 451 (Marubozu)

## The claim under test

- **The folklore.** A *marubozu* ("bald head" / "shaven head" in Japanese) is a candle whose
  real body fills almost the entire high-low range — it has (essentially) no wicks/shadows. A
  **bullish** marubozu opens at the low and closes at the high; a **bearish** one opens at the
  high and closes at the low. The lore reads it as a sign of **decisive, one-way pressure that
  continues**: a bullish marubozu is a high-probability buy. This is a retail/technician staple
  taught on TradingView, Investopedia, StockCharts' ChartSchool and every candlestick primer.
- **The source.** Japanese candlestick charting is traditionally attributed to the rice trader
  **Munehisa Homma** (18th century); the technique was introduced to Western markets and
  systematised by **Steve Nison**, *Japanese Candlestick Charting Techniques* (1991), and
  *Beyond Candlesticks* (1994), which name and illustrate the marubozu and its continuation
  reading. Gregory Morris, *Candlestick Charting Explained* (1992/2006) catalogues the same
  patterns with frequency tables.
- **Variants.** "Opening" and "closing" marubozu (one wick allowed) and the bearish marubozu are
  affine relatives of the same body-fill geometry and inherit the same drift confound tested
  here.

## Why this is a mechanical-proxy study

The marubozu is unusually *objective* for a candlestick pattern — it is a pure function of one
bar's OHLC — so we encode it directly with explicit thresholds (body ≥ 95% of range, each wick
≤ 2% of range) and state the only free parameters explicitly:

- **Objective detection.** Body and wick fractions are computed from the bar's own OHLC; the
  signal is fully known at the bar's close (no look-ahead in detection).
- **One execution lag.** The long is entered at the **next** close — the documented lag — so the
  signal bar's own (large, positive) return is never harvested.
- **The honest baseline.** The only meaningful comparison on an upward-drifting index is the
  **random-entry** control (same instrument, epoch and hold), because *any* long entry inherits
  the drift. We add a **body-shuffle placebo** that re-assigns the marubozu label to random bars
  (same count, same price marginal) — the direct test of "does the no-wick body matter?".

## Why a high one-sample t (or a high win-rate) is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample *t*
  of a long-only entry rule against **zero**, or a >50% win-rate, measures that drift, not the
  rule. See Fama & French on the equity premium; the desk's standing rule is *signal-vs-baseline*,
  never *signal-vs-zero*. (Here even the one-sample *t* is flat — the marubozu is too rare to pick
  up much beta.)
- **Data snooping on chart patterns.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis*, Journal of Finance) formalise testing chart/candlestick patterns against a properly
  matched null; Sullivan, Timmermann & White (1999, *Data-Snooping, Technical Trading Rule
  Performance, and the Bootstrap*, JF) and White (2000, *A Reality Check for Data Snooping*,
  Econometrica) show how pattern-fitted rules manufacture significance unless raced against a fair
  benchmark. Marshall, Young & Rose (2006, *Candlestick technical trading strategies: Can they
  create value for investors?*, Journal of Banking & Finance) test candlestick patterns directly
  and find **no value** beyond chance — consistent with this teardown.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the marubozu-vs-random difference.

## Method lineage (the desk's shared engine)

- **Candle geometry + body-fill rule.** [`strategy.candle_parts`](../marubozu/strategy.py),
  [`strategy.is_bullish_marubozu`](../marubozu/strategy.py) — body/wick fractions, the wickless
  threshold.
- **Forward-return + HAC t + random baseline.** [`strategy.forward_returns`](../marubozu/strategy.py),
  [`strategy.hac_t`](../marubozu/strategy.py), [`strategy.run_experiment`](../marubozu/strategy.py).
- **Geometry placebo.** [`strategy.body_shuffle_placebo`](../marubozu/strategy.py) — relabel
  random bars as "marubozu", keep the count and the marginal.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../marubozu/data.py) plants a
  real marubozu-continuation (knob `edge`); with `edge = 0` the detector must NOT manufacture
  significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../../450-andrews-pitchfork`](../../450-andrews-pitchfork) — the sibling "chart-geometry
  forecasts" teardown with the same random-entry + geometry-placebo idiom.
- [`../../402-`](../../) and the broader candlestick/technical-indicator zoo (hammers, dojis,
  engulfings, stars, soldiers) — most land None × Mirage for the same reason: a single bar's
  shape re-describes the day it happened, it does not forecast the next.
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting) frame why a
  signal-vs-zero *t* (or a >50% win-rate) is not enough; the marubozu is a clean live example of a
  vivid pattern with no forecasting content.
