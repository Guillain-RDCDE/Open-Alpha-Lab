# References & literature map — Study 462 (Rising Wedge)

## The claim under test

- **The folklore.** A *rising wedge* is two converging **up-sloping** trendlines: a support
  line through rising swing lows and a resistance line through rising swing highs, with the
  lower line climbing *faster* so the channel narrows toward an apex. Textbook technical
  analysis labels it a **bearish reversal/continuation** pattern: price is "supposed" to
  resolve to the **downside**, breaking *down* through the rising support. The retail rule —
  built into every chart-pattern site, TradingView screener and trading course — is **short the
  lower-line break**. (Its mirror, the *falling wedge*, is called bullish.)
- **The source.** The wedge is one of the classical chart patterns codified by **Robert D.
  Edwards & John Magee**, *Technical Analysis of Stock Trends* (1948 and later editions), the
  founding text of pattern-based TA, building on Richard Schabacker's *Technical Analysis and
  Stock Market Profits* (1932). The modern reference data is **Thomas N. Bulkowski**,
  *Encyclopedia of Chart Patterns* (2005), who catalogues rising-wedge "break-out" directions
  and (notably) reports that the rising wedge breaks *down* only modestly more often than a coin
  flip and that performance ranks poorly among patterns — a quantitative caveat the folklore
  drops. John Murphy's *Technical Analysis of the Financial Markets* and StockCharts'
  ChartSchool restate the bearish-wedge rule.
- **Variants.** "Rising wedge", "ascending wedge" and "rising broadening wedge" differ only in
  the slope/convergence convention; all are **affine variants of the same two-rising-line
  geometry** and inherit the same drift confound tested here.

## Why this is a "theory" / mechanical-proxy study

The wedge is *semi-subjective*: a discretionary trader chooses which swings to connect and how
many touches qualify a line. Following the desk's design for this kind, we encode the **tightest
mechanical rule a proponent would accept** and state the irreducible subjectivity explicitly:

- **Objective pivots.** Confirmed **fractals** (Bill Williams' fractal definition: a local
  extremum with *k* strictly-lower/higher bars on each side), only usable *k* bars later — a
  documented confirmation lag, no look-ahead.
- **Objective wedge.** Least-squares support/resistance lines through the last confirmed
  lows/highs; the pattern qualifies only when both rise, support rises faster (converging), and
  the lines have not yet crossed. No hand-picking of touches.
- **The honest baseline.** The only meaningful comparison on an upward-drifting index is the
  **random-entry** control (same instrument, epoch, hold, *and the same short sign*), because a
  short inherits the drift as a loss regardless of the pattern. We add a **slope-scramble
  placebo** that randomizes the support line while keeping the break cadence and price marginal —
  the direct test of "does the wedge's geometry matter?"

Hand-drawn wedges add *hindsight* (which swings, which touches — free parameters), which can only
inflate in-sample fit; the mechanical version is therefore the charitable **upper bound** on the
method.

## Why a significant signal-vs-zero t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample *t*
  of a **short** rule against **zero** measures that drift working *against* the short, not the
  pattern. A large negative one-sample *t* here is the drift, not skill. The desk's standing rule
  is *signal-vs-baseline*, never *signal-vs-zero*. See Fama & French on the equity premium.
- **Data snooping on chart patterns.** **Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis*, Journal of Finance)** formalize testing chart patterns (including triangles and
  wedges, via kernel-smoothed pivots) against a properly matched null and find most add little
  conditional information. **Sullivan, Timmermann & White (1999, *Data-Snooping, Technical
  Trading Rule Performance, and the Bootstrap*, JF)** and **White (2000, *A Reality Check for
  Data Snooping*, Econometrica)** show how pattern-fitted rules manufacture significance unless
  raced against a fair benchmark and corrected for the universe of rules searched.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the break-vs-random difference.

## Method lineage (the desk's shared engine)

- **Confirmed-fractal pivots + rolling wedge.** [`strategy.find_pivots`](../rising_wedge/strategy.py),
  [`strategy.build_wedges`](../rising_wedge/strategy.py) — the mechanical geometry with the
  confirmation lag baked in.
- **Short forward-return + HAC t + random baseline.** [`strategy.forward_returns`](../rising_wedge/strategy.py)
  (short sign), [`strategy.hac_t`](../rising_wedge/strategy.py),
  [`strategy.run_experiment`](../rising_wedge/strategy.py).
- **Geometry placebo.** [`strategy.slope_scramble_placebo`](../rising_wedge/strategy.py) —
  randomize the support slope/level, keep the break cadence and price marginal.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../rising_wedge/data.py) plants
  a real rising-wedge break-down (knob `edge`: rising/narrowing build-up then a bearish apex);
  with `edge = 0` the detector must NOT manufacture significance — the offline core runs with no
  network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced
  by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../450-andrews-pitchfork`](../450-andrews-pitchfork) — the sibling "price respects these
  lines" chart-geometry study built on the same confirmed-fractal-pivot engine and random-entry
  baseline idiom; same None × Mirage outcome.
- [`../414-falling-wedge`](../414-falling-wedge) — the *bullish* mirror of this pattern, and
  [`../411-ascending-triangle`](../411-ascending-triangle), [`../412-symmetrical-triangle`](../412-symmetrical-triangle)
  and the broader chart-figure zoo — most land None × Mirage for the same reason: a pattern
  fitted to past price re-describes the trend rather than forecasting the next move.
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting) frame why a
  significant signal-vs-zero *t* is not enough; the rising wedge is a clean live example of beta
  (here working *against* a short) masquerading as a bearish chart pattern.
