# References & literature map — Study 415 (Triple Top & Bottom)

## The claim under test

- **The folk recipe.** The triple top and triple bottom are canonical **reversal** figures in
  classical charting. Richard Schabacker (*Technical Analysis and Stock Market Profits*, 1932) and
  then Robert Edwards & John Magee (*Technical Analysis of Stock Trends*, 1948 — the field's
  founding text, still in print) codified them: three swing highs (or lows) at roughly one price
  level — *"three failures to break through"* — define an exhausted trend, and a confirmed close
  through the **neckline** (the most extreme intervening pivot) signals the reversal. The triple
  is sold as **more reliable than the double** precisely because the level has been defended three
  times. We steelman this as: *forward returns after a mechanically-confirmed triple-bottom
  breakout (and triple-top breakdown) beat the name's own base rate, net of costs, on daily equity
  bars — and the figure reverses symmetrically on both sides.*

## Why the steelman is *almost* coherent — the real ideas it leans on

- **Support/resistance as memory.** There is a genuine micro-structure rationale for horizontal
  levels: limit orders, round numbers, and anchoring can cluster supply/demand at a price (e.g.
  the disposition effect, Shefrin & Statman 1985, parks selling pressure near prior highs). A level
  *defended* repeatedly could, in principle, mark where that pressure is exhausted.
- **Short-horizon reversal exists — elsewhere.** Jegadeesh (1990) and Lehmann (1990) document
  one-month reversal at the single-stock level; De Bondt & Thaler (1985) document long-horizon
  (3–5y) reversal. A reversal *figure* might proxy these — but the triple's 1–8-week window sits in
  the ambiguous zone between reversal and momentum, and the breakout *entry* is a continuation bet,
  not a reversal one.
- **Bulkowski's catalogue.** Thomas Bulkowski (*Encyclopedia of Chart Patterns*, 2005) tabulates
  "success rates" for the triple top/bottom — but on a hand-curated sample without a base-rate
  control or a multiple-testing correction, exactly the selection problem this study controls for.

## The failure mode exposed

- **Subjectivity → selection by eye.** "Three taps at one level" admits enormous discretion; a
  pattern recognised only after the breakout is curve-fittable. Our mechanical detector removes the
  hindsight, and the strictness sweep shows the (non-)edge is not a single-tolerance artefact.
- **The up-drift base rate.** Any long signal on equities inherits the market's positive drift, so
  the *raw* post-breakout return looks positive. Netting out each name's base rate and racing the
  result against a **same-tape random-date placebo** strips the illusion — and the figure lands
  inside the luck cloud (*t* < 2, placebo *p* ≈ 0.37).
- **The asymmetry tell.** A real reversal figure must work on both sides. The bearish **triple
  top** short is *net-negative* here — the cleanest possible evidence that the bullish side was
  borrowed market drift, not exhaustion at a level. This echoes the broad "technical figures don't
  survive an honest test" literature: Brock, Lakonishok & LeBaron (1992) and especially Sullivan,
  Timmermann & White (1999), *"Data-Snooping, Technical Trading Rule Performance, and the
  Bootstrap"* (Journal of Finance), and Lo, Mamaysky & Wang (2000), *"Foundations of Technical
  Analysis"* (Journal of Finance) — which builds the *only* serious algorithmic pattern detector in
  the literature and finds head-and-shoulders/triple figures carry little to no incremental
  information once conditioned properly. Park & Irwin (2007), *"What Do We Know About the
  Profitability of Technical Analysis?"* (Journal of Economic Surveys), survey the wreckage.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *"A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix"* (Econometrica) —
  [`strategy.hac_t`](../triple_top_bottom/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Same-tape label-shuffle / random-date placebo.** The honest null for a selected event set on a
  drifting tape — [`strategy.run_experiment`](../triple_top_bottom/strategy.py).
- **Reproducibility stamp.** [`quantlab/repro.py`](../../../quantlab/repro.py) — the as-of freeze
  and content fingerprint each headline run carries.

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), auto-adjusted (split + dividend) OHLC across
  SPY + 29 long-listed US large-caps, 2005–2026 (as-of 2026-05-31). The offline reproducible core
  and the synthetic control run on the deterministic
  [`data.synthetic_panel`](../triple_top_bottom/data.py) generator, never the network. The headline
  is pinned with an as-of date and a content fingerprint (see [`docs/results.md`](results.md)).

## Related desk studies

- **[Study 411 — Ascending Triangle](../../411-ascending-triangle/)**: the same mechanical-detector
  + base-rate + same-tape-placebo machinery on a *continuation* figure — the direct sibling.
- **[Study 410 — Cup and Handle](../../410-cup-and-handle/)**, **[412 — Symmetrical Triangle](../../412-symmetrical-triangle/)**,
  **[413 — Bull Flag](../../413-bull-flag/)**, **[414 — Falling Wedge](../../414-falling-wedge/)**:
  the rest of the chart-figure cohort, same honest treatment.
- **[Study 178 — CCI](../../178-cci/)** and **[Study 104 — Bollinger-Reversion](../../104-bollinger-reversion/)**:
  oscillator "overbought/oversold reversal" rules — the indicator cousins of the reversal-figure family.
