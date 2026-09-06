# Sources & literature map — Study 991 (The Slow Bell)

## The stylised fact

- **Cont, R. (2001), "Empirical Properties of Asset Returns: Stylized Facts and Statistical
  Issues", *Quantitative Finance* 1(2), 223-236.** The canonical list, in which "aggregational
  Gaussianity" appears — and where the caveat that convergence is slow is stated but rarely
  quoted.
- **Mandelbrot, B. (1963), "The Variation of Certain Speculative Prices", *Journal of Business*
  36(4), 394-419.** The original claim that returns follow a stable law with infinite variance —
  in which case the convergence this study measures would never complete.
- **Fama, E. F. (1965), "The Behavior of Stock-Market Prices", *Journal of Business* 38(1),
  34-105.** The empirical follow-up, and the start of the sixty-year argument about the tail
  index.

## The tail index, which decides whether the theorem applies

- **Hill, B. M. (1975), "A Simple General Approach to Inference About the Tail of a
  Distribution", *Annals of Statistics* 3(5), 1163-1174.** The estimator in `hill_estimator`.
- **Jansen, D. W. & de Vries, C. G. (1991), "On the Frequency of Large Stock Returns", *Review of
  Economics and Statistics* 73(1), 18-24.** Estimates the equity tail index between 3 and 5 —
  above 2, so the variance exists, but below or near 4, so the kurtosis may not.
- **Loretan, M. & Phillips, P. C. B. (1994), "Testing the Covariance Stationarity of
  Heavy-Tailed Time Series", *Journal of Empirical Finance* 1(2), 211-248.** Why the sample
  kurtosis of a series with α < 4 is not an estimate of anything.

## Why aggregation is slower than i.i.d.

- **Bollerslev, T. (1986), "Generalized Autoregressive Conditional Heteroskedasticity",
  *Journal of Econometrics* 31(3), 307-327.** The model whose temporal aggregation properties
  explain the slowdown.
- **Drost, F. C. & Nijman, T. E. (1993), "Temporal Aggregation of GARCH Processes",
  *Econometrica* 61(4), 909-927.** The exact result: aggregated GARCH is GARCH with a slower
  kurtosis decay than the i.i.d. rate. The theoretical backbone of section 2.
- **Diebold, F. X. (1988), *Empirical Modeling of Exchange Rate Dynamics*, Springer.** Early
  documentation that conditional heteroskedasticity, not tail fatness, drives the slow
  convergence.

## Testing normality, and failing to

- **Jarque, C. M. & Bera, A. K. (1980), *Economics Letters* 6(3), 255-259**, and **Anderson,
  T. W. & Darling, D. A. (1954), *JASA* 49(268), 765-769.** The two tests used, chosen because
  they weight the tails differently.
- **Thadewald, T. & Büning, H. (2007), "Jarque-Bera Test and its Competitors for Testing
  Normality", *Journal of Applied Statistics* 34(1), 87-105.** The power comparison that
  motivates section 4.

## Neighbours on this desk

**311-fat-tails**, **427-return-distributions**, **256-volatility-clustering**,
**990-var-breach-count**, **970-annualisation-factors**.
