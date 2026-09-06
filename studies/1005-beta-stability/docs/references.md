# Sources & literature map — Study 1005 (Beta Has a Half-Life)

## Persistence and shrinkage

- **Blume, M. E. (1971), "On the Assessment of Risk", *Journal of Finance* 26(1), 1-10.** The
  persistence regression reproduced in section 1, and the origin of shrinking toward one.
- **Blume, M. E. (1975), "Betas and Their Regression Tendencies", *Journal of Finance* 30(3),
  785-795.** Argues the regression tendency is partly real — firms' projects mean-revert in
  risk — rather than purely statistical. Section 8's control is the attempt to separate the two.
- **Vasicek, O. A. (1973), "A Note on Using Cross-Sectional Information in Bayesian Estimation
  of Security Betas", *Journal of Finance* 28(5), 1233-1239.** Precision-weighted shrinkage,
  implemented as `vasicek_shrink`.
- **Klemkosky, R. C. & Martin, J. D. (1975), "The Adjustment of Beta Forecasts", *Journal of
  Finance* 30(4), 1123-1128.** An early out-of-sample horse race between adjustment schemes —
  the same exercise as section 6.
- **Elton, E. J., Gruber, M. J. & Urich, T. J. (1978), "Are Betas Best?", *Journal of Finance*
  33(5), 1375-1384.** Finds simple forecasts competitive with elaborate ones, which is the
  spirit of the `always_one` baseline.

## Estimation error

- **Fama, E. F. & MacBeth, J. D. (1973), "Risk, Return, and Equilibrium: Empirical Tests",
  *Journal of Political Economy* 81(3), 607-636.** Portfolio grouping to reduce beta measurement
  error — the logic behind section 3.
- **Dimson, E. (1979), "Risk Measurement When Shares Are Subject to Infrequent Trading",
  *Journal of Financial Economics* 7(2), 197-226.** The non-synchronous trading bias noted in
  the caveats.
- **Scholes, M. & Williams, J. (1977), "Estimating Betas from Nonsynchronous Data", *Journal of
  Financial Economics* 5(3), 309-327.**

## Time-varying beta

- **Bollerslev, T., Engle, R. F. & Wooldridge, J. M. (1988), "A Capital Asset Pricing Model with
  Time-Varying Covariances", *Journal of Political Economy* 96(1), 116-131.**
- **Jagannathan, R. & Wang, Z. (1996), "The Conditional CAPM and the Cross-Section of Expected
  Returns", *Journal of Finance* 51(1), 3-53.**
- **Lewellen, J. & Nagel, S. (2006), "The Conditional CAPM Does Not Explain Asset-Pricing
  Anomalies", *Journal of Financial Economics* 82(2), 289-314.** Argues beta variation is too
  small to matter for pricing — consistent with the decomposition in section 2.

## Neighbours on this desk

**1010-correlation-matrix-stability**, **1012-benchmark-choice-and-alpha**,
**240-low-beta-anomaly**, **998-kalman-hedge-ratio**.
