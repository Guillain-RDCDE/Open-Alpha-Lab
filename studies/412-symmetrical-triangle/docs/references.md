# References & literature map — Study 412 (Symmetrical Triangle)

## The claim under test

- **The folklore.** The symmetrical triangle — a sequence of *lower highs* and *higher lows*
  converging toward an apex — is taught as a **continuation** figure: price coils, then breaks
  out *in the prevailing trend's direction*, and the post-breakout move "runs about the height
  of the triangle." Buy the confirmed up-breakout, short the confirmed down-breakout. It is one
  of the canonical figures in every technical-analysis curriculum.
- **The technician canon.** Robert Edwards & John Magee, *Technical Analysis of Stock Trends*
  (1948 → 11th ed.) define the symmetrical triangle and the breakout/measured-move rule. Thomas
  Bulkowski, *Encyclopedia of Chart Patterns* (2nd ed., 2005) catalogues hit-rates and "measured
  move" targets for triangles. John Murphy, *Technical Analysis of the Financial Markets* (1999)
  is the standard reference for the converging-trendline construction.

## What the evidence actually says about chart patterns

- **The foundational test.** Andrew Lo, Harry Mamaysky & Jiang Wang, *Foundations of Technical
  Analysis* (2000, Journal of Finance) build *automated, kernel-smoothed* detectors for ten
  classic figures (head-and-shoulders, triangles, etc.) and test conditional return
  distributions — the first rigorous, mechanical pattern study and the methodological ancestor
  of this one. They find some statistical content in a few patterns but weak economic value.
- **Triangles specifically underperform.** Bulkowski's own data and subsequent replications find
  symmetrical-triangle breakouts close to a coin flip once the up/down base rate (market drift)
  is accounted for. Carol Osler & others on support/resistance, and the broad chart-pattern
  literature, repeatedly show that *visible* trendline figures carry little out-of-sample edge.
- **Why mechanical matters.** David Aronson, *Evidence-Based Technical Analysis* (2006) and the
  data-snooping critiques (below) show that subjective chart reading is unfalsifiable: only a
  pre-specified mechanical rule can be tested honestly. We test the closest mechanical definition
  and say explicitly that a hand-drawn triangle could differ.

## Why a high raw *t* still needs a base rate (the trap we expose)

- **Beta over a long window.** Over 60 trading days the broad market drifts up; a "long"
  bucket rises and a "short" bucket falls *regardless* of the signal. The cure is a **matched
  random-day base rate** that carries the same drift, so the *excess* isolates the figure's
  contribution. Our up/down 60-day split (+3.9% / −3.4%) is exactly this artefact — it dies in
  the excess column.
- **Newey-West HAC** standard errors (Whitney Newey & Kenneth West, 1987, *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*,
  Econometrica) for the excess *t*; overlapping forward windows induce autocorrelation a naive
  *t* ignores.
- **Data snooping.** Halbert White (2000, *A Reality Check for Data Snooping*, Econometrica) and
  Harvey, Liu & Zhu (2016, *…and the Cross-Section of Expected Returns*, RFS): a figure mined
  from thousands of charts will look good by luck. The permutation placebo here is the
  small-effect null in that spirit.

## Method lineage (the desk's shared engine)

- **Mechanical figure detector.** [`strategy.detect_triangles`](../symmetrical_triangle/strategy.py)
  — swing pivots via `scipy.signal.find_peaks`, least-squares trendline fits, a converging +
  symmetric-slope + range-contraction filter, and a price-pierce breakout confirmation.
- **Random-day placebo + HAC *t* on the excess.**
  [`strategy.run_pooled`](../symmetrical_triangle/strategy.py) and
  [`strategy.summarize`](../symmetrical_triangle/strategy.py) — the same idiom as the
  double-top study (189): signal vs matched random-day base rate, HAC *t* on the excess.
- **Permutation placebo.** [`strategy.permutation_p`](../symmetrical_triangle/strategy.py) —
  20,000 label shuffles of the pooled signal/base returns.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../symmetrical_triangle/data.py)
  splices in real triangles and plants a known post-breakout continuation; with the edge at zero
  the inference must NOT manufacture significance. The offline core runs with no network.

## Data sources used here

- **yfinance** daily OHLCV (`auto_adjust=True`) for a fixed 30-name large-cap basket incl. SPY,
  2005-01-03 → 2026-06-18, cached as parquet under `_cache/`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [189 · Double-Top / Double-Bottom](../../189-double-top/) — the sibling reversal-figure
  teardown; same random-day placebo + HAC idiom.
- [104 · Bollinger-Reversion](../../104-bollinger-reversion/) — a band-pierce study with the same
  "does the signal beat a random entry?" question.
- The **research-method demos** (data-mining-roulette, multiple-testing, look-ahead) frame why a
  big raw *t* on a long-horizon bucket is usually market beta, not edge — this study is a live
  example.
