# Sources & literature map — Study 978 (The Resampled Frontier)

## The method

- **Michaud, R. O. (1998), *Efficient Asset Management*.** The book that introduced resampled
  efficiency, and the patent that followed it.
- **Michaud, R. O. & Michaud, R. (2008), "Estimation Error and Portfolio Optimization: A
  Resampling Solution", *Journal of Investment Management* 6(1), 8-28.** The clearest statement
  of the case, and the one this implementation follows.
- **Jorion, P. (1992), "Portfolio Optimization in Practice", *Financial Analysts Journal*
  48(1), 68-74.** The earlier resampling experiment that showed how wide the distribution of
  "optimal" portfolios really is.

## The critique

- **Scherer, B. (2002), "Portfolio Resampling: Review and Critique", *Financial Analysts
  Journal* 58(6), 98-109.** The standard objection: averaging optimal portfolios is not
  optimal, the procedure inherits the estimator's bias, and much of its benefit is an implicit
  constraint. The central comparison in this study is Scherer's argument, measured.
- **Harvey, C. R., Liechty, J. C. & Liechty, M. W. (2008), "Bayes vs. Resampling: A Rematch",
  *Journal of Investment Management* 6(1), 29-45.** A Bayesian treatment dominates resampling
  on its own terms.
- **Markowitz, H. M. & Usmen, N. (2003), "Resampled Frontiers versus Diffuse Bayes: An
  Experiment", *Journal of Investment Management* 1(4), 9-25.** The famous experiment that went
  the other way; read alongside Harvey et al.

## The competitors

- **Ledoit, O. & Wolf, M. (2003, 2004).** Covariance shrinkage — this desk's study **975**.
- **Jorion, P. (1986), "Bayes-Stein Estimation for Portfolio Analysis", *JFQA* 21(3),
  279-292.** Shrinking expected returns toward a grand mean; the crude version used here.
- **DeMiguel, V., Garlappi, L. & Uppal, R. (2009), *Review of Financial Studies* 22(5),
  1915-1953.** 1/N.
- **Jagannathan, R. & Ma, T. (2003), *Journal of Finance* 58(4), 1651-1683.** Why long-only is
  itself shrinkage — relevant because every portfolio here is long-only.

## Neighbours on this desk

**975-covariance-shrinkage**, **976-hierarchical-risk-parity**, **977-max-diversification**,
**979-black-litterman-zero-views**, **171-naive-1-over-n**, **968-bootstrap-choice**,
**967-rolling-vs-expanding**.
