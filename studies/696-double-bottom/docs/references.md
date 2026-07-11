# References & literature map — Study 696 (Double-Bottom)

## The claim under test

- **The folklore.** The **double bottom** — a "W" on the chart — is one of the oldest named
  reversal figures in classical charting: price falls to a **trough**, rallies to an intervening
  peak (the **neckline**), falls back to a **second trough at roughly the same level**, then
  **breaks and closes above the neckline**. The reading: the market tested a support level twice
  and twice failed to break it, so sellers are exhausted, buyers are in control, and the
  **measured-move target** — the trough-to-neckline height, projected up from the neckline — is
  where price is headed next. Richard Schabacker (*Technical Analysis and Stock Market Profits*,
  1932) and then Robert Edwards & John Magee (*Technical Analysis of Stock Trends*, 1948 — the
  field's founding text, still in print) codify it as a **major** reversal figure. Thomas
  Bulkowski's *Encyclopedia of Chart Patterns* (2005/2021) catalogs measured "success rates" and
  measured-move statistics for it on a hand-curated sample; this study measures the figure
  independently, with its own mechanical detector, basket and honest controls, and does not
  borrow his numbers.
- **What we test.** Whether entering long the day after a confirmed double-bottom breakout earns
  a return that beats the same stock's own base rate over 1/5/10/20/40-day horizons; whether the
  classic measured-move target is touched more often than a magnitude-matched random walk; and
  whether a "long timer" — hold to target-or-timeout — nets a real excess after costs.

## Why the detector needs an honest, mechanical definition

- **Chart figures are partly in the eye of the beholder.** "Two troughs at a similar level" is
  loose enough that three chartists will draw three different W's on the same tape. We write
  down the closest **mechanical** rule we can — two swing lows within a tolerance of one level,
  separated by a **genuine** intervening rally (not a flat shelf), then a confirmed close through
  the neckline — and say so loudly (see [`strategy.detect_double_bottom`](../double_bottom/strategy.py)).
  A robustness sweep over the tolerance and rally-size parameters is reported so the reader can
  see the (non-)edge does not depend on one lucky choice of knob.
- **The up-drift base rate.** Any long signal on equities inherits the market's own positive
  drift, so the *raw* post-breakout return looks positive almost by construction. Netting out
  each name's own base rate and racing the result against a **same-tape random-date placebo**
  strips the illusion.
- **The measured-move target needs a matched control, not a bare hit rate.** A target's hit rate
  in isolation says little — any target close enough to the entry price will get touched often on
  pure noise. We draw a **magnitude-matched placebo**: random entries on the same tapes given a
  target at the *same* relative distance as the real signals' average target, so "does the target
  beat a coin flip of the same size" is answered honestly (see
  [`strategy.measured_move_hits`](../double_bottom/strategy.py)).
- **The long timer needs a matched base rate too.** Holding to target-or-timeout produces a
  variable holding period per trade; comparing its P&L to *zero* rewards nothing but the market's
  own drift over that (multi-week-average) holding window. We race it against a
  **holding-period-matched** base rate instead (see [`strategy.timer_pnl`](../double_bottom/strategy.py)).

## The broad evidence on chart-pattern reversal claims

- **Lo, Mamaysky & Wang (2000), *Foundations of technical analysis* (Journal of Finance)** — the
  only serious algorithmic pattern-detection study in the literature; finds *some* patterns carry
  marginal statistical information on 1962–1996 US data, but the double-bottom/-top family is not
  among the clean survivors once selection and costs are addressed.
- **Marshall, Young & Rose (2006), *Candlestick technical trading strategies: Can they create
  value for investors?* (Journal of Banking & Finance)** and **Marshall, Young & Cahan (2008)** —
  test the broad chart/candlestick taxonomy and find no value once data-snooping is accounted
  for; our large-cap null is consistent with this literature.
- **Sullivan, Timmermann & White (1999), *Data-snooping, technical trading rule performance, and
  the bootstrap* (Journal of Finance)** — the canonical demonstration that a universe of technical
  rules, tested without correction, inflates apparent significance; motivates this study's
  same-tape placebo and robustness sweep.
- **Park & Irwin (2007), *What do we know about the profitability of technical analysis?*
  (Journal of Economic Surveys)** — a survey of the broader wreckage.

## Method lineage (the desk's shared engine)

- **Mechanical swing-pivot detector + event study.**
  [`strategy.swing_pivots`](../double_bottom/strategy.py) and
  [`strategy.detect_double_bottom`](../double_bottom/strategy.py) — the same swing-pivot
  machinery as siblings [415-triple-top-bottom](../../415-triple-top-bottom/) and
  [695-inverse-head-shoulders](../../695-inverse-head-shoulders/), adapted to a two-trough figure.
- **Drift-neutral inference vs the base rate + same-tape placebo.**
  [`strategy.run_experiment`](../double_bottom/strategy.py) — one-sample and HAC *t*, and a
  random-date placebo on the same tape, the same engine lineage as 415 and 695.
- **Measured-move hit rate with a magnitude-matched control, and a target-or-timeout long timer.**
  [`strategy.measured_move_hits`](../double_bottom/strategy.py) and
  [`strategy.timer_pnl`](../double_bottom/strategy.py) — the same two extra arbiters study 695
  runs for the inverse head-and-shoulders' measured move, applied here to the simpler two-trough
  figure.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../double_bottom/data.py) plants a
  known post-breakout drift on a panel with its own embedded up-drift; with the edge set to zero
  the placebo must NOT manufacture significance (checked over 20 seeds) even though a single-seed
  naive *t* can be misleadingly positive — the offline core runs with no network.

## Data sources used here

- **yfinance** daily **auto-adjusted** (split + dividend) OHLC for a fixed 30-name liquid
  large-cap + SPY basket — 2005-01-03 → 2026-06-30, cached under `_cache/db_*.parquet`. All
  headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- **[189-double-top](../../189-double-top/)** — the closest relative and the study's namesake
  pair: 189's detector *also* finds double-tops **and** double-bottoms, but on a fixed-horizon
  forward-return-vs-random-day-placebo protocol with a Bonferroni correction across pattern types
  and horizons, no measured-move target and no long timer. This study (696) is the dedicated
  bullish double-bottom study with its own mechanical neckline/breakout geometry (a genuine
  intervening rally is required, not just "the close broke a local trough's high"), a
  base-rate-neutral excess (not a raw-return comparison), a **measured-move hit-rate test with a
  magnitude-matched placebo**, and a **target-or-timeout long timer with a holding-period-matched
  base rate** — none of which 189 runs. Both studies land on the same honest conclusion
  (Signal = NONE) via different, complementary machinery — a nice cross-check, not a duplicate.
- **[415-triple-top-bottom](../../415-triple-top-bottom/)** — the three-tap version of the same
  "tested a level and failed" idea. Requires a genuine third tap (and, for the bearish myth-check,
  a mirror-image triple top); this study's figure is the simpler, more common **two**-trough case
  and does not test a triple anywhere.
- **[695-inverse-head-shoulders](../../695-inverse-head-shoulders/)** — the three-trough version
  with an asymmetric middle trough (the "head," strictly the deepest of the three) and two
  shoulders of similar, shallower depth. This study's two troughs are symmetric by construction —
  there is no head, and no shoulder-symmetry test — a structurally different figure that shares
  only the swing-pivot/neckline/measured-move machinery, not the detector's geometry.
- **[694-matching-low](../../694-matching-low/)** — the **micro**, two-*candle* version of "tested
  the same price twice": two consecutive **down** candles whose **closes** (not swing-pivot lows)
  match within a tight tolerance, confirmed the very next session. This study is the **macro**,
  multi-week **swing-chart** version spanning many bars between the two troughs, detected on
  swing pivots rather than adjacent candles, with a neckline breakout rather than an implied
  next-day bounce. The two figures can co-occur on the same names but test different geometry at
  different time scales; neither study runs the other's detector.

None of the siblings run this study's specific two-trough, base-rate-neutral, measured-move
+ long-timer bar — the double-bottom's tradability, as distinct from its mere existence, is this
study's own axis.
