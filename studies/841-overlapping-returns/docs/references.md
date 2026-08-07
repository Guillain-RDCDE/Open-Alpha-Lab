# References & literature map — Study 841 (Overlapping-Returns Inflation)

## The claim, at full strength

For decades the empirical case for return predictability rested on **long-horizon predictive
regressions**: forecast the cumulative return over the next 1, 3, 5 or 10 years from a valuation ratio
(dividend yield, earnings yield, CAPE) known today, sampled monthly or quarterly. The R² rose and the
t-statistics grew with the horizon — seemingly compelling evidence that "the long run is
predictable". But because monthly-sampled `h`-period returns *overlap* — adjacent observations share
`h−1` months of returns — the regression residuals are a moving average of order `h−1`, and the
ordinary-least-squares standard errors that ignore this are **grossly understated**. Much of the
apparent long-horizon predictability is a mechanical artefact of the overlap. This study makes the
trap undeniable by running the regression on a world we *built* to have **zero predictability**, so
any long-horizon t-stat or R² above the nominal level is, by construction, spurious.

## The source papers — overlapping data and long-horizon inference

- **Hansen, L. P. & Hodrick, R. J. (1980)**, *"Forward Exchange Rates as Optimal Predictors of Future
  Spot Rates: An Econometric Analysis."* *Journal of Political Economy* 88(5). The origin of the
  Hansen-Hodrick GMM standard errors for regressions with **overlapping** observations — the
  autocorrelation-consistent covariance that the overlap requires.
- **Hodrick, R. J. (1992)**, *"Dividend Yields and Expected Stock Returns: Alternative Procedures for
  Inference and Measurement."* *Review of Financial Studies* 5(3). The source paper for this study.
  Hodrick shows that overlapping long-horizon regressions have severely over-sized test statistics
  and proposes the **"1B" standard error** — reformulating the regression so the summation falls on
  the *regressor* (giving non-overlapping one-period moments) — which has far better finite-sample
  size. This is the correction implemented in `strategy.hodrick_1b_slope_t`.
- **Newey, W. K. & West, K. D. (1987)**, *"A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix."* *Econometrica* 55(3). The HAC (Bartlett-kernel)
  sandwich estimator; with `lags ≈ h−1` it absorbs the induced MA(h−1) structure. The second
  correction in this study.

## The finite-sample evidence — how bad, and which fix wins

- **Ang, A. & Bekaert, G. (2007)**, *"Stock Return Predictability: Is it There?"* *Review of
  Financial Studies* 20(3). A careful re-examination that relies on **Hodrick (1992) standard
  errors** precisely because overlapping long-horizon OLS t-stats are untrustworthy; much of the
  long-horizon evidence weakens under honest inference.
- **Boudoukh, J., Richardson, M. & Whitelaw, R. F. (2008)**, *"The Myth of Long-Horizon
  Predictability."* *Review of Financial Studies* 21(4). Shows that long-horizon regression
  coefficients and R²s are *mechanically* related to the one-period ones under persistence, so the
  rising R² with horizon is largely an artefact — the empirical companion to this demo's finding.
- **Wei, M. & Wright, J. H. (2013)**, *"Reverse Regressions and Long-Horizon Forecasting."* *Journal
  of Applied Econometrics* 28(3). A modern comparison of overlapping-regression inference procedures
  (Hodrick 1B, Newey-West, reverse regressions), confirming the Hodrick estimator's superior size —
  the ranking this study reproduces by Monte Carlo.
- **Valkanov, R. (2003)**, *"Long-Horizon Regressions: Theoretical Results and Applications."*
  *Journal of Financial Economics* 68(2). Asymptotic theory for the non-standard behaviour of
  long-horizon regressions when the horizon grows with the sample — why naive t-stats do not converge
  to the usual distribution.

## The persistent-regressor cousin

- **Stambaugh, R. F. (1999)**, *"Predictive Regressions."* *Journal of Financial Economics* 54(3).
  The finite-sample bias that arises when a **persistent** predictor's innovation is correlated with
  the return innovation (the `delta` in this study's DGP). Distinct from — but often compounding —
  the overlap problem; the data-generating process here uses the Stambaugh form so the demonstration
  is faithful to the real predictors (valuation ratios) that motivated the literature.

## Neighbours on this bench (the dedup map)

- **[Study 838 — HAC-Necessity](../../838-hac-necessity/)** — HAC standard errors on a *trading
  strategy's own daily P&L* (the autocorrelation of a rule's realised returns). Study 841 is the
  **predictive-regression** cousin: the autocorrelation is induced by *overlapping the dependent
  variable*, not by the strategy's own return dynamics, and the fix of record is Hodrick's 1B
  estimator rather than a plain daily HAC.
- **[Study 835 — Spurious-Regression](../../835-spurious-regression/)** — the Granger-Newbold
  spurious regression between two independent **unit-root / trending** series (a non-stationarity
  mechanism). Study 841's predictor and returns are **stationary**; the inflation here comes purely
  from the overlap of cumulative returns, a different trap.
- **[Study 346 — Multiple-Testing](../../346-multiple-testing/)** — inflated significance from
  testing **many hypotheses**, corrected by a trial-count haircut. Study 841 inflates a **single**
  hypothesis via serial correlation, corrected by an autocorrelation-consistent covariance — a
  different source of false significance and a different fix.

## Shared method

- **The Hodrick (1992) 1B standard error** — the overlap-robust covariance that builds its moments
  from non-overlapping one-period returns; the best-sized of the three tests in every Monte Carlo
  here.
- **Newey-West (1987) HAC** — the Bartlett-kernel sandwich, `lags = h−1`, the general-purpose but
  finite-sample-imperfect correction.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a synthetic
  control is a machinery proof, never market evidence; `REAL` needs a robust *t* ≥ 2 on a real tape —
  which a synthetic-only demo can never provide), and the ≥ 20-seed / large-Monte-Carlo rule for any
  synthetic-dependent claim.
