# Sources & literature map — Study 975 (Shrink the Matrix)

## The estimator

- **Ledoit, O. & Wolf, M. (2003), "Improved Estimation of the Covariance Matrix of Stock
  Returns with an Application to Portfolio Selection", *Journal of Empirical Finance* 10(5),
  603-621.** The constant-correlation target tested here, and the paper that made shrinkage
  practical for equity portfolios.
- **Ledoit, O. & Wolf, M. (2004), "A Well-Conditioned Estimator for Large-Dimensional
  Covariance Matrices", *Journal of Multivariate Analysis* 88(2), 365-411.** The identity
  target and the analytic intensity formula.
- **Ledoit, O. & Wolf, M. (2017), "Nonlinear Shrinkage of the Covariance Matrix for Portfolio
  Selection", *Review of Financial Studies* 30(12), 4349-4388.** The successor that dominates
  the linear version — the natural fork of this study.
- **Stein, C. (1956), "Inadmissibility of the Usual Estimator for the Mean of a Multivariate
  Normal Distribution", *Proc. Third Berkeley Symposium*.** Where the whole idea starts: in
  high dimensions, the obvious estimator is beatable.

## Why the optimiser makes it worse

- **Michaud, R. O. (1989), "The Markowitz Optimization Enigma: Is 'Optimized' Optimal?",
  *Financial Analysts Journal* 45(1), 31-42.** Error maximisation, named.
- **Jobson, J. D. & Korkie, B. (1980), "Estimation for Markowitz Efficient Portfolios",
  *JASA* 75(371), 544-554.** The size of the estimation problem, quantified early.
- **Jagannathan, R. & Ma, T. (2003), "Risk Reduction in Large Portfolios: Why Imposing the
  Wrong Constraints Helps", *Journal of Finance* 58(4), 1651-1683.** A no-short constraint *is*
  shrinkage — the comparison run in this study's long-only section.
- **DeMiguel, V., Garlappi, L. & Uppal, R. (2009), "Optimal Versus Naive Diversification",
  *Review of Financial Studies* 22(5), 1915-1953.** The benchmark that beats most of this:
  1/N. Any shrinkage study that does not mention it is selling something.

## Random matrix theory, the other route to the same fix

- **Marchenko, V. A. & Pastur, L. A. (1967).** The limiting spectrum of a sample covariance
  matrix — why the eigenvalue spread is predictable rather than mysterious.
- **Laloux, L., Cizeau, P., Bouchaud, J.-P. & Potters, M. (1999), "Noise Dressing of Financial
  Correlation Matrices", *Physical Review Letters* 83(7), 1467-1470.** The physics literature's
  version: most of an equity correlation matrix's eigenvalues are indistinguishable from noise.

## Neighbours on this desk

**967-rolling-vs-expanding**, **976-hierarchical-risk-parity**, **977-max-diversification**,
**978-resampled-frontier**, **171-naive-1-over-n**, **890-sector-risk-parity**,
**973-calendar-misalignment** (a different way to get a correlation matrix wrong).
