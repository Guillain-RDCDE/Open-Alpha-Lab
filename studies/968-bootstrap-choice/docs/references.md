# Sources & literature map — Study 968 (Which Bootstrap)

## The methods

- **Efron, B. (1979), "Bootstrap Methods: Another Look at the Jackknife", *Annals of
  Statistics* 7(1), 1-26.** The original i.i.d. resample.
- **Kunsch, H. R. (1989), "The Jackknife and the Bootstrap for General Stationary
  Observations", *Annals of Statistics* 17(3), 1217-1241.** The moving-block bootstrap: resample
  blocks, keep the local dependence.
- **Politis, D. N. & Romano, J. P. (1992), "A Circular Block-Resampling Procedure for Stationary
  Data"**, in *Exploring the Limits of Bootstrap*. Wrapping the series so that every observation
  is drawn equally often — the fix for the moving block's end effects, and this desk's default.
- **Politis, D. N. & Romano, J. P. (1994), "The Stationary Bootstrap", *JASA* 89(428),
  1303-1313.** Geometric block lengths; the resampled series is genuinely stationary.
- **Politis, D. N. & White, H. (2004), "Automatic Block-Length Selection for the Dependent
  Bootstrap", *Econometric Reviews* 23(1), 53-70.** The data-driven block length this study
  deliberately does not use, so that the sweep can show why it matters.

## Standard errors for a Sharpe ratio

- **Lo, A. W. (2002), "The Statistics of Sharpe Ratios", *Financial Analysts Journal* 58(4),
  36-52.** The autocorrelation correction; implemented in `quantlab.analytics.sharpe_with_se`.
- **Mertens, E. (2002), "Comments on Variance of the IID Estimator in Lo (2002)".** The
  higher-moment (skew/kurtosis) correction — the right analytic answer for fat-tailed returns.
- **Christie, S. (2005), "Is the Sharpe Ratio Useful in Asset Allocation?"** and
  **Opdyke, J. D. (2007), *Journal of Asset Management* 8, 308-336.** Further work on the
  Sharpe's sampling distribution under non-i.i.d. returns.

## Why any of this matters for backtests

- **Harvey, C. R. & Liu, Y. (2015), "Backtesting", *Journal of Portfolio Management* 42(1),
  13-28**, and **Bailey, D. H. & Lopez de Prado, M. (2014), "The Deflated Sharpe Ratio",
  *Journal of Portfolio Management* 40(5), 94-107.** Both take the standard error as an input;
  if the interval under-covers, everything built on it inherits the error.
- **Ledoit, O. & Wolf, M. (2008), "Robust Performance Hypothesis Testing with the Sharpe
  Ratio", *Journal of Empirical Finance* 15(5), 850-859.** The studentised bootstrap for Sharpe
  differences — the natural next step from this study.

## Neighbours on this desk

**841-overlapping-returns**, **838-hac-necessity**, **833-deflated-sharpe-ratio**,
**834-minimum-backtest-length**, **840-clustered-standard-errors**, **346-multiple-testing**,
**839-tstat-three-threshold**.
