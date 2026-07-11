# References & literature map — Study 706 (Diamond Bottom)

## The claim under test

- **The folklore.** A **diamond bottom** is the bullish mirror of the diamond top: the swing
  range first **broadens** (a megaphone / broadening formation — successively higher highs
  and lower lows) and then **narrows** (a symmetrical triangle — lower highs and higher
  lows), so the price envelope traces a diamond, this time forming **after a decline**. The
  lore says the diamond marks *accumulation* at a low, and the **upside breakout** out of the
  narrowing apex confirms a reversal — so you **buy the breakout**, often with a measured-move
  target equal to the diamond's height. It is taught, like the diamond top, as one of the
  rarer but "high-reliability" reversal patterns.
- **The source.** The broadening formation and the symmetrical triangle were catalogued by
  **Robert D. Edwards & John Magee**, *Technical Analysis of Stock Trends* (1948 and later
  editions) — the foundational chart-pattern text; the diamond is the broadening-then-
  symmetrical composite, and Edwards & Magee explicitly note it can appear as either a top or
  a bottom formation. **Thomas N. Bulkowski**, *Encyclopedia of Chart Patterns* (2nd ed.,
  2005) gives the modern statistical write-up, separately tabulating diamond tops and diamond
  bottoms. StockCharts' ChartSchool, Investopedia and every charting suite restate the rule.
- **Variants.** The "diamond top" (same shape, downside break, a bearish reversal — see
  [study 466](../../466-diamond-top/)) is the direct mirror; the looser "broadening bottom /
  megaphone" without the narrowing leg ([study 465](../../465-broadening-formation/)) is an
  affine relative of the same swing-amplitude geometry and inherits the same drift confound.

## Why this is a "theory" / mechanical-proxy study

The diamond bottom is *semi-subjective*: a discretionary chartist decides which swings count
and when the shape is "complete." Following the desk's design for this kind (identical
protocol to study 466, mirrored for the long side), we encode the **tightest mechanical rule
a proponent would accept** and state the irreducible subjectivity explicitly:

- **Objective pivots.** Confirmed **fractals** (a local extremum with *k* strictly-lower/
  higher bars on each side, Bill Williams' fractal definition), only usable *k* bars later — a
  documented confirmation lag, no look-ahead.
- **Objective diamond.** Over the 6 most-recent alternating confirmed pivots, the leg
  amplitudes must *increase* to a peak (broadening) then *decrease* (narrowing), with the
  diamond formed after a **decline** (the trough is reached *during* the run, not sitting at
  its very start); no hand-picking which swings to connect.
- **The honest baseline.** The only meaningful comparison is the **random-entry** control
  (same instrument, epoch, hold and long side), because a long on an upward-drifting index
  banks the drift no matter when it fires — a naive one-sample *t* against zero would flatter
  *any* bottom-shaped rule for free. We add a **shuffled-pivot placebo** that destroys the
  broaden-then-narrow geometry while keeping the price marginal — the direct test of "does the
  diamond shape matter?"

Hand-anchored diamonds add *hindsight* (a free parameter), which can only inflate in-sample
fit; the mechanical version is therefore the charitable **upper bound** on the method.

## Why the one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample
  *t* of a **long-only** rule against **zero** is flattered by that drift regardless of
  whether the pattern means anything — the mirror image of the diamond-top study, where the
  same drift punishes a short for free. The desk's standing rule is *signal-vs-baseline*,
  never *signal-vs-zero*. See Fama & French on the equity premium.
- **Data snooping on chart patterns.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis*, Journal of Finance) formalize testing chart patterns (including head-and-shoulders
  and triangles) against a properly matched null and find little out-of-sample value; Sullivan,
  Timmermann & White (1999, *Data-Snooping, Technical Trading Rule Performance, and the
  Bootstrap*, JF) and White (2000, *A Reality Check for Data Snooping*, Econometrica) show how
  shape-fitted rules manufacture significance unless raced against a fair benchmark.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the breakout-vs-random difference.

## Method lineage (the desk's shared engine)

- **Confirmed-fractal pivots + diamond geometry.** [`strategy.find_pivots`](../diamond_bottom/strategy.py),
  [`strategy.is_diamond`](../diamond_bottom/strategy.py),
  [`strategy.diamond_breakouts`](../diamond_bottom/strategy.py) — the mechanical broaden-then-
  narrow detector with the confirmation lag baked in, entirely mirroring
  [study 466](../../466-diamond-top/)'s `diamond_breakdowns` with the context check and the
  breakout direction flipped for a bottom.
- **Forward-return (long) + HAC t + random baseline.** [`strategy.forward_returns`](../diamond_bottom/strategy.py),
  [`strategy.hac_t`](../diamond_bottom/strategy.py), [`strategy.run_experiment`](../diamond_bottom/strategy.py).
- **Geometry placebo.** [`strategy.shuffled_pivot_placebo`](../diamond_bottom/strategy.py) —
  permute pivot prices, keep positions and marginals.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../diamond_bottom/data.py)
  plants a real diamond-bottom reversal (knob `edge`, a sell-off then a broaden/narrow
  diamond then a genuine rally); with `edge = 0` the detector must NOT manufacture
  significance across seeds — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-06-30 (As-of 2026-06-30, the last complete calendar month), cached as
  parquet under `_cache/`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies — the dedup map (what this study is NOT)

- [466-diamond-top](../../466-diamond-top/) — the **bearish mirror**: same geometry, a
  short on the downside break after an advance. That study finds the short *loses* to a
  drift-matched random short (60d Welch *t* = −2.55, the wrong way) and the shape placebo
  clears it (*p* = 0.68). This study runs the identical engine on the **long, after-a-decline**
  side — the honest question is whether the mirror flip changes anything, or whether it is
  just as flat once raced against a random long (it is: see [`docs/results.md`](results.md)).
- [465-broadening-formation](../../465-broadening-formation/) — the diamond's *broadening*
  leg alone, no narrowing apex required, tested as a short on the lower-boundary break. A
  looser, one-sided relative of the same swing-amplitude geometry — also None × Mirage.
- [695-inverse-head-shoulders](../../695-inverse-head-shoulders/) — a different bullish
  reversal figure (three troughs, a neckline break, a measured-move target) with its own
  detector; shares the "chart figure meets an honest baseline" idiom but a different
  geometry entirely — that study's breakout also fails to beat the stock's own base rate
  (best HAC *t* = 1.56).
- [705-rounding-top](../../705-rounding-top/) — the bearish "saucer" distribution top,
  a smooth-curvature figure rather than a broaden/narrow swing pattern; a different detector,
  the same honest-baseline discipline.
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting) frame why a
  signal-vs-zero *t* is not enough; the diamond bottom is a clean live example of a rare,
  eye-catching shape whose apparent "hit rate" is really just the market's own upward drift
  wearing a costume.
