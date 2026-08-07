# References & literature map — Study 838 (HAC Necessity)

## The claim under test — the source papers

- **The HAC estimator.** Newey, W. K. & West, K. D. (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica, 55(3),
  703–708). The foundational fix: a Bartlett-kernel-weighted sum of sample autocovariances that
  yields a consistent, positive-semidefinite long-run-variance estimate — the standard error the
  naive OLS *t* gets wrong on serially-correlated data. This is the estimator implemented in
  [`strategy.newey_west_t`](../hac_necessity/strategy.py) and validated against `statsmodels`.
- **Overlapping observations — where the autocorrelation comes from.** Hansen, L. P. & Hodrick,
  R. J. (1980), *Forward Exchange Rates as Optimal Predictors of Future Spot Rates* (Journal of
  Political Economy, 88(5)). The canonical treatment of overlapping-return regressions: a *K*-period
  overlap induces an MA(*K*−1) error structure, and the naive standard error must be inflated to
  account for it. Our hero generator (a trailing *K*-day rolling mean of i.i.d. innovations) is
  exactly this object, with the closed-form long-run-variance ratio *K*.

## The bandwidth question (how many lags)

- **Automatic bandwidth.** Newey, W. K. & West, K. D. (1994), *Automatic Lag Selection in Covariance
  Matrix Estimation* (Review of Economic Studies, 61(4)) — the data-driven rule; the plug-in
  `floor(4·(n/100)^(2/9))` used in [`strategy.nw_auto_lags`](../hac_necessity/strategy.py) is the
  common textbook default. Our AR(1) tail (ρ = 0.8) shows *why* the choice matters: too few lags and
  even the HAC *t* over-rejects.
- **Optimal kernels and prewhitening.** Andrews, D. W. K. (1991), *Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix Estimation* (Econometrica, 59(3)); Andrews & Monahan
  (1992). The Bartlett kernel we use is the simplest of a family; the quadratic-spectral kernel is
  MSE-optimal, and VAR-prewhitening reduces the residual over-rejection we document.
- **Finite-sample over-rejection.** Kiefer, N. M. & Vogelsang, T. J. (2005), *A New Asymptotic
  Theory for Heteroskedasticity-Autocorrelation Robust Tests* (Econometric Theory, 21). Explains the
  small residual over-rejection of standard HAC tests (our NW false-positive rate ~9.5%, not exactly
  5%) and the fixed-*b* alternative that improves size.

## Why this matters for return predictability

- **Long-horizon / overlapping predictive regressions.** Britten-Jones, M., Neuberger, A. &
  Nolte, I. (2011), *Improved inference in regression with overlapping observations* (Journal of
  Business Finance & Accounting); Boudoukh, Israel & Richardson (2022), *Biases in long-horizon
  predictive regressions* — the modern statements that overlap-inflated *t*-stats are a leading cause
  of spurious "predictability."
- **Textbook treatment.** Cochrane, J. H. (2005), *Asset Pricing* (Princeton), ch. 20, and Campbell,
  Lo & MacKinlay (1997), *The Econometrics of Financial Markets* — the standard references for HAC
  standard errors on autocorrelated asset returns.

## Method lineage (the desk's shared engine)

- **Multiple-testing cousins.** The desk's [Study 346 — multiple-testing](../../346-multiple-testing/)
  corrects for *how many hypotheses* you tried; this study corrects the *single* hypothesis's
  standard error for autocorrelation. Both are ways an un-adjusted *t* > 2 lies.
- **Overlapping-returns cousin.** [Study 841 — overlapping-returns](../../841-overlapping-returns/)
  works the same MA(*K*−1) structure from the returns-construction side; Study 838 is the inference
  side of the same coin (the standard error, not the return object).
- **Curve-fitting cousin.** [Study 348 — curve-fitting](../../348-curve-fitting/) shows a different
  route to a fake edge (fitting flexibility), where 838 shows the mis-specified variance.

## Data

- **None — this is a simulation study.** Every number is produced by the deterministic seeded
  generators in [`data.py`](../hac_necessity/data.py) (an overlapping-window MA process and an AR(1)
  process); there is no market data, no network call, and no cache. The headline run is pinned by the
  config fingerprint `767e2ce61be1` and the null-matrix content fingerprint `0c98419fb4d7` (as-of
  2026-06-30). See [`docs/results.md`](results.md).
