# Sources & literature map — Study 967 (Window Shopping)

## Why means are hopeless and second moments are not

- **Merton, R. C. (1980), "On Estimating the Expected Return on the Market", *Journal of
  Financial Economics* 8(4), 323-361.** The foundational result: the precision of an estimated
  mean depends on the *calendar span*, not the sampling frequency, so no amount of daily data
  fixes it — while variance estimates improve with frequency. This is the single most important
  reference for reading section 2.
- **Jorion, P. (1986), "Bayes-Stein Estimation for Portfolio Analysis", *Journal of Financial
  and Quantitative Analysis* 21(3), 279-292.** Shrinking means toward a grand mean; the
  benchmark used here.
- **Best, M. J. & Grauer, R. R. (1991), "On the Sensitivity of Mean-Variance-Efficient
  Portfolios to Changes in Asset Means", *Review of Financial Studies* 4(2), 315-342.** How
  small errors in means produce absurd portfolios.
- **DeMiguel, V., Garlappi, L. & Uppal, R. (2009), "Optimal Versus Naive Diversification",
  *Review of Financial Studies* 22(5), 1915-1953.** Why 1/N beats optimisation once estimation
  error is honest — the backdrop to the covariance section.

## Beta and its drift

- **Blume, M. E. (1971), "On the Assessment of Risk", *Journal of Finance* 26(1), 1-10.** Betas
  regress toward one; the 2/3-1/3 shrinkage tested here.
- **Vasicek, O. A. (1973), "A Note on Using Cross-Sectional Information in Bayesian Estimation
  of Security Betas", *Journal of Finance* 28(5), 1233-1239.** The Bayesian version, shrinking
  by each estimate's own precision.
- **Fama, E. F. & MacBeth, J. D. (1973), "Risk, Return, and Equilibrium: Empirical Tests",
  *Journal of Political Economy* 81(3), 607-636.** The rolling-window estimation convention
  that the profession inherited.

## Covariance matrices and error maximisation

- **Michaud, R. O. (1989), "The Markowitz Optimization Enigma: Is 'Optimized' Optimal?",
  *Financial Analysts Journal* 45(1), 31-42.** Optimisers maximise estimation error; the
  "optimism" column in section 3 is this effect, measured.
- **Ledoit, O. & Wolf, M. (2003), "Improved Estimation of the Covariance Matrix of Stock
  Returns with an Application to Portfolio Selection", *Journal of Empirical Finance* 10(5),
  603-621.** The shrinkage answer to the short-window problem — this desk's study **975**.
- **Bai, J. & Perron, P. (1998), "Estimating and Testing Linear Models with Multiple Structural
  Changes", *Econometrica* 66(1), 47-78.** The other side of the trade-off: if parameters
  genuinely break, a long window is a liability.
- **Pesaran, M. H. & Timmermann, A. (2007), "Selection of Estimation Window in the Presence of
  Breaks", *Journal of Econometrics* 137(1), 134-161.** The paper that asks exactly this
  study's question, formally.

## Comparing the estimators

- **Diebold, F. X. & Mariano, R. S. (1995), *JBES* 13(3), 253-263**, and **Newey & West (1987),
  *Econometrica* 55(3), 703-708** — the paired-loss comparison and its HAC standard error.

## Neighbours on this desk

**975-covariance-shrinkage**, **971-sqrt-time-scaling**, **968-bootstrap-choice**,
**1008-beta-stability**, **836-timing-luck**, **171-naive-1-over-n**, **902-multi-factor-composite**.
