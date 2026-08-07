# References & literature map — Study 840 (Clustered Standard Errors)

## The claim under test — the source papers

- **The panel-data survey (the source paper).** Petersen, M. A. (2009), *Estimating Standard
  Errors in Finance Panel Data Sets: Comparing Approaches* (Review of Financial Studies, 22(1),
  435–480). The definitive demonstration that finance panels carry **two** kinds of residual
  dependence — a persistent **firm effect** (correlation *within a firm, over time*) and a
  common **time effect** (correlation *across firms, within a period*) — and that OLS, White,
  and *one-way* clustered standard errors are badly biased whenever you cluster on the wrong
  dimension or ignore the dependence. His Monte-Carlo tables are exactly the experiment
  reproduced here for the **time-effect** case: with a common time shock the naive OLS SE is far
  too small and firm-clustering does not help; you must cluster by time (or use Fama-MacBeth).
- **The Fama-MacBeth estimator.** Fama, E. F. & MacBeth, J. D. (1973), *Risk, Return, and
  Equilibrium: Empirical Tests* (Journal of Political Economy, 81(3), 607–636). Run a
  cross-sectional regression **each period**, then take the mean and the *time-series* standard
  error of the period-by-period slopes. Because the *T* per-period estimates are the independent
  units, the procedure is automatically robust to arbitrary cross-sectional (within-period)
  correlation — the correction implemented in [`strategy.panel_inference`](../clustered_se/strategy.py).
- **The variance-inflation closed form.** Moulton, B. R. (1986), *Random Group Effects and the
  Precision of Regression Estimates* (Journal of Econometrics, 32(3)); Moulton (1990), *An
  Illustration of a Pitfall in Estimating the Effects of Aggregate Variables on Micro Units*
  (Review of Economics and Statistics). The "Moulton factor" √(1 + (N̄−1)·ρ_x·ρ_e): when a
  regressor with intra-group correlation ρ_x meets a residual with intra-group correlation ρ_e
  in groups of size N̄, the true SE exceeds the naive SE by exactly this factor. It is the
  ground truth our simulated naive-*t* SD is matched against.

## Clustering — how many dimensions, how many clusters

- **Two-way / multi-way clustering.** Cameron, A. C., Gelbach, J. B. & Miller, D. L. (2011),
  *Robust Inference with Multiway Clustering* (Journal of Business & Economic Statistics, 29(2)),
  and Thompson, S. B. (2011), *Simple formulas for standard errors that cluster by both firm and
  time* (Journal of Financial Economics, 99(1)). The general fix when **both** a firm effect and
  a time effect are present: cluster on both. For the pure time effect isolated here, two-way
  clustering coincides with time clustering and with Fama-MacBeth.
- **Few-cluster problems.** Cameron, A. C. & Miller, D. L. (2015), *A Practitioner's Guide to
  Cluster-Robust Inference* (Journal of Human Resources, 50(2)); Bertrand, Duflo & Mullainathan
  (2004), *How Much Should We Trust Differences-in-Differences Estimates?* (QJE, 119(1)).
  Cluster-robust SEs are asymptotic in the *number of clusters*; with few clusters they
  over-reject — visible in our N = 2 firm-clustering row (39.8%) and the mild time-clustering
  over-rejection (7.4%) at only T = 50 time clusters.

## Why this matters for cross-sectional asset pricing

- **Cross-sectional return predictability.** Cochrane, J. H. (2005), *Asset Pricing* (Princeton),
  ch. 12 and 20; Campbell, Lo & MacKinlay (1997), *The Econometrics of Financial Markets* — the
  standard references for Fama-MacBeth inference on cross-sectional return regressions, precisely
  because a common market/time factor makes pooled OLS *t*-stats untrustworthy.
- **The replication-crisis backdrop.** Harvey, C. R., Liu, Y. & Zhu, H. (2016), *…and the
  Cross-Section of Expected Returns* (Review of Financial Studies, 29(1)) — many published
  "factors" carry *t*-stats inflated by exactly the dependence corrections (multiple testing,
  clustering, HAC) this desk isolates one at a time.

## Method lineage (the desk's shared engine — the dedup map)

- **HAC cousin (the *time-series* dependence).** [Study 838 — HAC Necessity](../../838-hac-necessity/)
  corrects a *single serially-correlated series* for autocorrelation *over time* (Newey-West).
  Study 840 is the **cross-sectional** analogue: correlation *across firms within a period*,
  corrected by time-clustering / Fama-MacBeth. Same disease (a mis-specified variance), a
  different axis of dependence.
- **Multiple-testing cousin (the *how-many-hypotheses* problem).** [Study 346 —
  multiple-testing](../../346-multiple-testing/) corrects a *t* for *how many hypotheses you
  tried*; Study 840 corrects a *single* hypothesis's standard error for cross-sectional
  dependence. Both are ways a naked *t* > 2 lies.
- Together these three isolate the three leading reasons an un-adjusted panel *t*-stat overstates
  significance: **too many trials** (346), **autocorrelation over time** (838), and
  **cross-sectional dependence within a period** (840, this study).

## Data

- **None — this is a simulation study.** Every number is produced by the deterministic seeded
  generator in [`data.py`](../clustered_se/data.py) (a null panel with a common time factor in
  both the regressor and the residual, plus a `beta` knob for the positive control); there is no
  market data, no network call, and no cache. The headline run is pinned by the config
  fingerprint `a271c7ebce63` and the null-panel content fingerprint `607a6862117f` (as-of
  2026-06-30). See [`docs/results.md`](results.md); reproduce with
  [`examples/verify.py`](../examples/verify.py).
