# References & literature map — Study 411 (Ascending Triangle)

## The claim under test

- **The folklore.** "An ascending triangle is a **bullish continuation** pattern: the price taps a
  **flat horizontal resistance** repeatedly while its lows **rise** toward it; demand absorbs the
  supply at that level and price eventually **breaks out upward**, often by about the height of the
  triangle. Buy the breakout." It is one of the most-taught figures in technical analysis.
- **The seminal source.** Robert D. Edwards & John Magee, *Technical Analysis of Stock Trends*
  (1948) — the founding text of charting — classifies the ascending triangle among the bullish
  continuation figures and prescribes the breakout-with-volume entry. Later popularisers (Schabacker
  before them; Bulkowski's *Encyclopedia of Chart Patterns*, 2000, after) carry the same recipe.
- **The systematic counter-evidence.** Thomas Bulkowski's own measured statistics show wide
  variation and modest, cost-sensitive edges; Andrew Lo, Harry Mamaysky & Jiang Wang,
  *Foundations of Technical Analysis* (2000, Journal of Finance) build the first rigorous
  kernel-smoothing detector for chart patterns and find only weak, mostly informational content.
  David Aronson, *Evidence-Based Technical Analysis* (2006) is the methodological warning shot: most
  charted patterns fail once you apply a proper null and correct for data-snooping.

## What we measure, and why the base-rate excess

- **Excess over the name's base rate.** A breakout selects a *date*, and equities drift up, so the
  raw forward return is dominated by the equity risk premium. We subtract each name's own
  unconditional mean forward return so the test is "does the figure beat buy-and-hold *for that
  name*?" — the only version of the question that isn't answered "yes" by construction.
- **The same-tape placebo.** Even the excess can be lifted by *momentum*: the breakout fires after a
  run-up, and recently-rising names keep rising a bit. The honest null is therefore **random entry
  dates on the same tape** (same count, same base-rate subtraction), which inherits that drift. This
  is the arbiter the naive *t* misses — and the one that demotes the verdict to WEAK here.
- **The down-break symmetry test.** The cleanest within-study control: run the *identical* detector
  but require a break **below** the rising floor. A figure that genuinely "breaks upward" must have
  its bearish resolution underperform; if both resolutions drift up the same, the direction carries
  no information. (This is the analogue of the breakout-vs-reversion contrast in the desk's
  Bollinger study and the gap-vs-surprise myth-check in PEAD.)
- **Timing / no look-ahead.** The breakout is known at its close; we enter the **next** close (one
  documented lag) and hold 5/10/20/40 days — the standard event-study convention.

## Why a high *t* still isn't a REAL stamp here

- **Drifting-tape false positives.** A one-sample/HAC *t* (Newey-West, 1987, for autocorrelation
  robustness) on post-breakout excess clears 2 — but a *synthetic zero-edge control* reproduces the
  same *t* purely from the planted shape's geometry, while the placebo correctly refuses to fire.
  The desk's inference bar (METHODOLOGY → *The inference bar*) reserves REAL for a robust *t* the
  honest null survives; here the null (placebo) busts it.
- **Selection on a famous rule.** Harvey, Liu & Zhu (2016, *…and the Cross-Section of Expected
  Returns*, RFS) and Lopez de Prado on backtest overfitting: a naive *t* near 2 on one of dozens of
  charted patterns is exactly what data-snooping manufactures. The placebo and the symmetry control
  are the corrections.

## Method lineage (the desk's shared engine)

- **Swing-pivot detector + breakout rule.**
  [`strategy.detect_triangles`](../ascending_triangle/strategy.py) — flat-top taps + rising-low
  trendline + first confirmed close above the rim (or below the floor for the myth-check).
- **Excess-over-base-rate + one-sample/HAC t.**
  [`strategy.run_experiment`](../ascending_triangle/strategy.py),
  [`strategy.hac_t`](../ascending_triangle/strategy.py).
- **Same-tape label-shuffle placebo.** Built into `run_experiment` — random entry dates on the same
  tape, the honest drifting-tape null.
- **Deterministic synthetic control.**
  [`data.synthetic_panel`](../ascending_triangle/data.py) plants clean triangles + a known
  post-breakout drift; with the edge set to zero the placebo must NOT fire — the offline core runs
  with no network.

## Data sources used here

- **yfinance** daily auto-adjusted OHLC for a fixed 30-name large-cap basket incl. SPY,
  2005-01-03 → 2026-05-29, as-of **2026-05-31**, cached under `_cache/triangle_{open,high,low,close}.parquet`.
  All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../410-cup-and-handle`](../../410-cup-and-handle) — the sibling chart-figure teardown (O'Neil's
  pivot); same base-rate-excess + same-tape-placebo method, same MIRAGE landing.
- [`../104-bollinger-reversion`](../../104-bollinger-reversion) — the breakout-vs-reversion symmetry
  idea that inspired this study's down-break control.
- [`../178-cci`](../../178-cci) and the broader technical-indicator zoo — most classic TA lands
  None/Weak × Mirage once the honest null is applied; this is another data point.
- The **research-method demos** (data-mining-roulette, multiple-testing, backtest-overfitting) frame
  why a *t* alone is not enough — the ascending triangle is a textbook example of a naive *t* clearing
  2 on a drifting tape and being busted by the placebo.
