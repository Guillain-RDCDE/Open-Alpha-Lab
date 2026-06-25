# References & literature map — Study 466 (Diamond Top)

## The claim under test

- **The folklore.** A **diamond top** is a reversal pattern that forms after an advance: the
  swing range first **broadens** (a megaphone / broadening formation — successively higher
  highs and lower lows) and then **narrows** (a symmetrical triangle — lower highs and higher
  lows), so the price envelope traces a diamond. The lore says the diamond marks distribution
  at a top, and the **downside breakout** out of the narrowing apex confirms a reversal — so
  you **short the breakdown**, often with a measured-move target equal to the diamond's
  height. It is taught as one of the rarer but "high-reliability" reversal patterns.
- **The source.** The broadening formation and the symmetrical triangle were catalogued by
  **Robert D. Edwards & John Magee**, *Technical Analysis of Stock Trends* (1948 and later
  editions) — the foundational chart-pattern text; the diamond is the broadening-then-
  symmetrical composite. **Thomas N. Bulkowski**, *Encyclopedia of Chart Patterns* (2nd ed.,
  2005) gives the modern statistical write-up (break direction, performance, failure rates).
  StockCharts' ChartSchool, Investopedia and every charting suite restate the rule.
- **Variants.** "Diamond bottom" (same shape, upside break, a bullish reversal), and the
  looser "broadening top / megaphone" without the narrowing leg, are affine relatives of the
  same swing-amplitude geometry and inherit the same drift confound tested here.

## Why this is a "theory" / mechanical-proxy study

The diamond top is *semi-subjective*: a discretionary chartist decides which swings count and
when the shape is "complete." Following the desk's design for this kind, we encode the
**tightest mechanical rule a proponent would accept** and state the irreducible subjectivity
explicitly:

- **Objective pivots.** Confirmed **fractals** (a local extremum with *k* strictly-lower/
  higher bars on each side, Bill Williams' fractal definition), only usable *k* bars later — a
  documented confirmation lag, no look-ahead.
- **Objective diamond.** Over the 6 most-recent alternating confirmed pivots, the leg
  amplitudes must *increase* to a peak (broadening) then *decrease* (narrowing), with the
  diamond formed after an advance; no hand-picking which swings to connect.
- **The honest baseline.** The only meaningful comparison is the **random-entry** control
  (same instrument, epoch, hold and *short sign*), because a short on an upward-drifting index
  loses the drift no matter when it fires. We add a **shuffled-pivot placebo** that destroys
  the broaden-then-narrow geometry while keeping the price marginal — the direct test of "does
  the diamond shape matter?"

Hand-anchored diamonds add *hindsight* (a free parameter), which can only inflate in-sample
fit; the mechanical version is therefore the charitable **upper bound** on the method.

## Why the one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample
  *t* of a long-only rule against **zero** measures that drift, and for a *short* it manufactures
  a misleading *negative* number that has nothing to do with the pattern. The desk's standing
  rule is *signal-vs-baseline*, never *signal-vs-zero*. See Fama & French on the equity
  premium.
- **Data snooping on chart patterns.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis*, Journal of Finance) formalize testing chart patterns (including head-and-shoulders
  and triangles) against a properly matched null and find little out-of-sample value; Sullivan,
  Timmermann & White (1999, *Data-Snooping, Technical Trading Rule Performance, and the
  Bootstrap*, JF) and White (2000, *A Reality Check for Data Snooping*, Econometrica) show how
  shape-fitted rules manufacture significance unless raced against a fair benchmark.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the break-vs-random difference.

## Method lineage (the desk's shared engine)

- **Confirmed-fractal pivots + diamond geometry.** [`strategy.find_pivots`](../diamond_top/strategy.py),
  [`strategy.is_diamond`](../diamond_top/strategy.py),
  [`strategy.diamond_breakdowns`](../diamond_top/strategy.py) — the mechanical broaden-then-
  narrow detector with the confirmation lag baked in.
- **Forward-return (short) + HAC t + random baseline.** [`strategy.forward_returns`](../diamond_top/strategy.py),
  [`strategy.hac_t`](../diamond_top/strategy.py), [`strategy.run_experiment`](../diamond_top/strategy.py).
- **Geometry placebo.** [`strategy.shuffled_pivot_placebo`](../diamond_top/strategy.py) —
  permute pivot prices, keep positions and marginals.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../diamond_top/data.py)
  plants a real diamond-top reversal (knob `edge`); with `edge = 0` the detector must NOT
  manufacture significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced
  by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../450-andrews-pitchfork`](../../450-andrews-pitchfork) — the sibling chart-geometry
  study (median-line channel) tested with the same random-entry + shuffled-pivot idiom; same
  None × Mirage verdict.
- [`../465-broadening-formation`](../../465-broadening-formation) (the diamond's broadening
  leg, standalone) and [`../188-head-shoulders`](../../188-head-shoulders) — the reversal
  chart-pattern family that re-describes a trend pause.
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting) frame why a
  signal-vs-zero *t* is not enough; the diamond top is a clean live example of a rare,
  eye-catching shape that carries no forecasting information once raced against a fair null.
