# References & literature map — Study 417 (Island Reversal)

## The claim under test

- **The folklore.** "When a trend exhausts, a final gap (the *exhaustion gap*) strands a small
  cluster of bars — the *island* — at the extreme. A second gap in the **opposite** direction
  (the *breakaway gap*) then seals the island off, leaving it visibly marooned in empty chart
  space. The two gaps bracketing the cluster mark a high-confidence **reversal**: short the
  island top, buy the island bottom, as soon as the sealing gap confirms it." It is one of the
  most visually striking figures in classical chart-reading.
- **The source.** Robert D. Edwards & John Magee, *Technical Analysis of Stock Trends* (1948 and
  its many editions) — the canonical text — describe the island reversal and the gap taxonomy
  (common / breakaway / runaway / exhaustion). Thomas N. Bulkowski, *Encyclopedia of Chart
  Patterns* (2nd ed., 2005) catalogues "island reversals, tops and bottoms" with measured
  performance statistics and is the modern reference for the figure's claimed reliability.
- **Why test it.** The island reversal is exactly the kind of figure that is *defined visually*
  ("a cluster marooned by two gaps") and then credited with predictive power. The honest question
  is whether the closest **mechanical** version carries any forward edge once you net out the
  underlying name's own drift — or whether, like most chart figures, it is a shape the eye finds
  in noise.

## What we measure, and why

- **An objective detector.** Chart figures are partly subjective, so we encode the closest
  rules-based definition: a gap of at least a threshold in one direction, a 1–3-bar island whose
  range never fills that first gap, then a sealing gap of the same size in the opposite direction
  that re-crosses the island's far edge *and* opens back inside the pre-gap range. We test that
  exact rule and say so — three chartists would draw the figure three ways, and a mechanical rule
  is the only reproducible test. (See [`strategy.detect_islands`](../island_reversal/strategy.py).)
- **Excess over the base rate.** A bullish figure on an asset that drifts up will "work" for free.
  We subtract each name's own unconditional mean forward return (direction-matched) so the test is
  "does the figure beat a *random* entry in the same direction on the same name", not "is the
  market up/down". This is the single most important honesty control for any pattern study.
- **The same-tape random-date placebo.** Beyond the *t*, we draw thousands of random entry dates
  on each name's own tape (same count, same horizon, same base-rate subtraction) and ask how often
  random beats the figure. For a drifting asset over multi-week holds the *t* can clear 2 on drift
  alone; the placebo is the arbiter that exposes that — Fisher's randomization logic (Efron &
  Tibshirani, *An Introduction to the Bootstrap*, 1993).
- **One execution lag, costs, HAC.** Signal known at the sealing gap's close; enter the **next**
  close (one documented lag); 5 bps one-way × 2 legs. The HAC (Newey-West, 1987) *t* guards
  against clustered islands inflating significance.

## Why "subjective figure → mechanical proxy" is the right frame

- **Lo, Mamaysky & Wang (2000, *Foundations of Technical Analysis*, JF)** built kernel-smoothed
  algorithmic detectors for head-and-shoulders and other figures — the methodological ancestor of
  encoding a chart figure as a reproducible rule and testing its conditional return distribution.
- **Bulkowski's own statistics** are computed on hand-identified figures and a survivorship-prone
  universe; an objective detector on a fixed basket with a base-rate net and a placebo is the
  stricter test. Our finding (no edge on the bearish side, drift-only on the bullish side) is
  consistent with the broad result that most classical chart figures do not survive an honest,
  base-rate-adjusted test.

## Method lineage (the desk's shared engine)

- **Objective detector + forward excess.** [`strategy.detect_islands`](../island_reversal/strategy.py)
  and [`strategy.run_experiment`](../island_reversal/strategy.py) — pooled per-event excess over the
  name's base rate.
- **Same-tape random-date placebo.** Built into `run_experiment` — the honest small-sample,
  drifting-tape null.
- **HAC / one-sample t.** [`strategy.hac_t`](../island_reversal/strategy.py) and
  [`strategy.one_sample_t`](../island_reversal/strategy.py).
- **Deterministic synthetic control.** [`data.synthetic_panel`](../island_reversal/data.py) plants
  clean island tops followed by a known post-island reversal; with the edge set to zero the
  inference must NOT manufacture significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily auto-adjusted OHLC for a fixed 30-name large-cap basket (SPY + 29), cached
  under `_cache/island_{open,high,low,close}.parquet`, 2005-01-03 → 2026-05-29 (as-of 2026-05-31).
  All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[415-triple-top-bottom](../../415-triple-top-bottom/)** — the sibling three-tap reversal figure,
  same objective-detector / base-rate-excess / placebo machinery; also NONE × MIRAGE, "reliable
  reversal" BUSTED.
- **[410-cup-and-handle](../../410-cup-and-handle/)**, **[411-ascending-triangle](../../411-ascending-triangle/)**,
  **[413-bull-flag](../../413-bull-flag/)** — the rest of the chart-figure family on this bench.
- **[363-pead-drift](../../363-pead-drift/)** — the *counter*-example: a folk effect that *does*
  clear the bar (REAL) once the fundamental surprise (not the visible gap) is the sort key — the
  contrast that shows the placebo/excess machinery can bank a genuine signal.
