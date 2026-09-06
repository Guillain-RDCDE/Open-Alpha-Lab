# Sources & literature map — Study 1010 (Mostly Noise)

## Random matrix theory

- **Marchenko, V. A. & Pastur, L. A. (1967), "Distribution of Eigenvalues for Some Sets of
  Random Matrices", *Mathematics of the USSR-Sbornik* 1(4), 457-483.** The band.
- **Laloux, L., Cizeau, P., Bouchaud, J.-P. & Potters, M. (1999), "Noise Dressing of Financial
  Correlation Matrices", *Physical Review Letters* 83(7), 1467-1470.** The application to equity
  correlation matrices, and the cleaning filter implemented as `rmt_clean`.
- **Plerou, V., Gopikrishnan, P., Rosenow, B., Amaral, L. A. N. & Stanley, H. E. (1999),
  "Universal and Nonuniversal Properties of Cross Correlations in Financial Time Series",
  *Physical Review Letters* 83(7), 1471-1474.** The independent simultaneous discovery.
- **Bouchaud, J.-P. & Potters, M. (2011), "Financial Applications of Random Matrix Theory: A
  Short Review", in *The Oxford Handbook of Random Matrix Theory*.** The survey, including the
  rotationally-invariant estimators that outperform the simple filter used here.
- **Bun, J., Bouchaud, J.-P. & Potters, M. (2017), "Cleaning Large Correlation Matrices: Tools
  from Random Matrix Theory", *Physics Reports* 666, 1-109.**

## Shrinkage

- **Ledoit, O. & Wolf, M. (2003), "Improved Estimation of the Covariance Matrix of Stock Returns
  with an Application to Portfolio Selection", *Journal of Empirical Finance* 10(5), 603-621.**
  The constant-correlation target implemented in `ledoit_wolf_shrink`.
- **Ledoit, O. & Wolf, M. (2004), "A Well-Conditioned Estimator for Large-Dimensional Covariance
  Matrices", *Journal of Multivariate Analysis* 88(2), 365-411.**
- **Ledoit, O. & Wolf, M. (2017), "Nonlinear Shrinkage of the Covariance Matrix for Portfolio
  Selection", *Review of Financial Studies* 30(12), 4349-4388.**

## Why it matters for portfolios

- **Michaud, R. O. (1989), "The Markowitz Optimization Enigma: Is 'Optimized' Optimal?",
  *Financial Analysts Journal* 45(1), 31-42.**
- **Jagannathan, R. & Ma, T. (2003), "Risk Reduction in Large Portfolios: Why Imposing the Wrong
  Constraints Helps", *Journal of Finance* 58(4), 1651-1683.** Shows a long-only constraint is
  equivalent to a form of shrinkage — the result section 8 reproduces empirically.
- **DeMiguel, V., Garlappi, L. & Uppal, R. (2009), "Optimal Versus Naive Diversification",
  *Review of Financial Studies* 22(5), 1915-1953.**
- **Chan, L. K. C., Karceski, J. & Lakonishok, J. (1999), "On Portfolio Optimization:
  Forecasting Covariances and Choosing the Risk Model", *Review of Financial Studies* 12(5),
  937-974.** The out-of-sample horse race format section 7 follows.

## Neighbours on this desk

**1005-beta-stability**, **1004-how-many-stocks**, **1003-bitcoin-in-a-portfolio**,
**1001-purged-cv-embargo**.
