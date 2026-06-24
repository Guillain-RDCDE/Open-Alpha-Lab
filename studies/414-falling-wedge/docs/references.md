# References & literature map — Study 414 (Falling Wedge)

## The claim under test

- **The folklore.** "A **falling wedge** — price drifting down inside two **downward-sloping,
  converging** trendlines — is a **bullish** figure. The selling exhausts itself as the range
  narrows; price then breaks **above** the upper line and runs. Buy the upside break." It is the
  bullish twin of the (bearish) rising wedge in the technician's toolbox, traded both as a reversal
  (after a downtrend) and a continuation (in an uptrend).
- **The textbook sources.** Robert Edwards & John Magee, *Technical Analysis of Stock Trends*
  (1948 and later editions) — the canonical catalogue of chart figures, including the wedge.
  Thomas Bulkowski, *Encyclopedia of Chart Patterns* (2nd ed., 2005) tabulates measured
  performance statistics for the falling wedge (break-out direction, average rise, failure rate)
  and is the most-cited modern reference believers point to.
- **Where it lives today.** The falling wedge is a staple of retail charting education
  (Investopedia, StockCharts ChartSchool, Bulkowski's ThePatternSite) and of trading-screen
  pattern scanners, which is exactly why an *objective, reproducible* test of the closest mechanical
  definition is worth doing.

## What we measure, and why a same-tape placebo is the right arbiter

- **An objective detector.** A falling wedge is partly subjective, so we encode the *closest
  mechanical* definition: a run of **descending** swing highs fit by a line, the intervening swing
  lows fit by a second line, **both slopes negative**, the **upper line steeper** than the lower
  (the highs fall faster — convergence), the vertical band **narrowing** by ≥25% toward an apex, and
  a confirmed close above the extrapolated upper line. Swing pivots are local extrema over a
  symmetric window (a pivot is only confirmable a few bars later — the detector respects that lag).
- **Excess over the base rate.** We subtract each name's own unconditional forward return, so the
  test is "does the figure beat buy-and-hold **for that name**", not "is the market up over
  2005–2026" — the latter is the trivially-true confound that flatters every long-only chart rule.
- **The same-tape label-shuffle placebo.** The decisive control. We draw random entry dates on the
  *same* names (same count, same base-rate subtraction) and ask how often a random set beats the
  observed breakout mean. Because the wedge breakout *selects* particular dates (post-selloff, on
  names that then keep grinding up), a naive *t*-against-zero inherits the basket's drift and
  overstates significance; the placebo holds that drift fixed. This is the Fisher randomization /
  White (2000) Reality-Check logic adapted to a single rule.
- **The down-break symmetry test.** If the figure genuinely chose an **upward** direction, the
  *down*-break of the identical geometry should drift the other way. Running both legs in one frame
  is the cleanest test of the directional premise.
- **One execution lag, costs both legs.** The breakout is known at its close; we enter the **next**
  close (one documented lag) and charge 5 bps one-way × 2 = 10 bps round trip.

## Why a high *t* still isn't a green stamp

- **One-sample / HAC *t*.** [`strategy.one_sample_t`](../falling_wedge/strategy.py) and
  [`strategy.hac_t`](../falling_wedge/strategy.py) — Newey-West (Bartlett-kernel) standard errors so
  temporally clustered breakouts don't inflate the statistic (Newey & West, 1987).
- **Selection / data-snooping.** Harvey, Liu & Zhu (2016, *…and the Cross-Section of Expected
  Returns*, RFS) and White (2000, *A Reality Check for Data Snooping*, Econometrica) on why a single
  *t* over a chosen rule overstates significance — the same-tape placebo is our reality check.
- **Chart-pattern evidence is mixed at best.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis*, JF) found *some* nonparametric-pattern informativeness but emphasised it is weak and
  regime-dependent; Bulkowski's own tables show high "failure" and "throwback" rates. Our WEAK ×
  MIRAGE landing is consistent with that literature once the drift confound is removed.

## Method lineage (the desk's shared engine)

- **Mechanical figure detector.** [`strategy.detect_wedges`](../falling_wedge/strategy.py) — swing
  pivots + dual descending-trendline fit + convergence + confirmed break.
- **Excess-over-base-rate event study + same-tape placebo.**
  [`strategy.run_experiment`](../falling_wedge/strategy.py) — pools per-event excess across the
  basket and arbitrates against random same-tape entry dates.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../falling_wedge/data.py) plants
  clean falling wedges with a *known* post-breakout continuation; with the edge set to zero the
  inference must NOT manufacture a positive edge — the offline core runs with no network.

## Data sources used here

- **yfinance** daily auto-adjusted OHLC for a fixed 30-name large-cap basket incl. SPY,
  2005-01-03 → 2026-05-29, cached under `_cache/wedge_{open,high,low,close}.parquet`. All headline
  numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **Sister chart-figure teardowns on this bench** — [411 ascending triangle](../../411-ascending-triangle/),
  [412 symmetrical triangle](../../412-symmetrical-triangle/), [413 bull flag](../../413-bull-flag/),
  [410 cup-and-handle](../../410-cup-and-handle/): the same objective-detector + same-tape-placebo
  protocol applied to the rest of the figure zoo. The recurring lesson is that the *direction* a
  figure claims to predict is the part that busts.
- The **research-method demos** (data-mining-roulette, multiple-testing, look-ahead) frame why a
  *t* alone is not enough — the falling wedge is a textbook case where the naive *t* clears 2 and the
  placebo says "tape drift."
