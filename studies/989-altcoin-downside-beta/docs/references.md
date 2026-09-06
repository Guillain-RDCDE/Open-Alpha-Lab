# Sources & literature map — Study 989 (The One-Way Beta)

## Downside beta, formally

- **Bawa, V. S. & Lindenberg, E. B. (1977), "Capital Market Equilibrium in a Mean-Lower Partial
  Moment Framework", *Journal of Financial Economics* 5(2), 189-200.** The LPM-CAPM and the
  downside beta implemented in `bawa_lindenberg_beta`.
- **Hogan, W. W. & Warren, J. M. (1974), "Toward the Development of an Equilibrium
  Capital-Market Model Based on Semivariance", *Journal of Financial and Quantitative Analysis*
  9(1), 1-11.** The semivariance beta in `hogan_warren_beta`.
- **Ang, A., Chen, J. & Xing, Y. (2006), "Downside Risk", *Review of Financial Studies* 19(4),
  1191-1239.** The paper that made downside beta a priced factor in equities, and the template
  this study follows for crypto.
- **Harvey, C. R. & Siddique, A. (2000), "Conditional Skewness in Asset Pricing Tests", *Journal
  of Finance* 55(3), 1263-1295.** Coskewness as a priced characteristic — the corroborating
  measurement in section 3.

## Why the naive test fails

- **Longin, F. & Solnik, B. (2001), "Extreme Correlation of International Equity Markets",
  *Journal of Finance* 56(2), 649-676.** Measured correlation changes in the tails *even under
  multivariate normality*. Section 4's simulated benchmark exists because of this paper.
- **Ang, A. & Chen, J. (2002), "Asymmetric Correlations of Equity Portfolios", *Journal of
  Financial Economics* 63(3), 443-494.** The exceedance-correlation framework and the correct
  way to test asymmetry against a null that already produces some.
- **Boyer, B. H., Gibson, M. S. & Loretan, M. (1999), "Pitfalls in Tests for Changes in
  Correlations", Federal Reserve IFS Discussion Paper 597.** The clearest statement of the
  conditioning bias: splitting on the regressor guarantees the appearance of a structural break.
- **Politis, D. N. & Romano, J. P. (1994), "The Stationary Bootstrap", *JASA* 89(428),
  1303-1313.** The block resampling in `asymmetry_test`.

## Crypto specifics

- **Liu, Y. & Tsyvinski, A. (2021), "Risks and Returns of Cryptocurrency", *Review of Financial
  Studies* 34(6), 2689-2727.** What prices the cross-section of crypto returns.
- **Borri, N. (2019), "Conditional Tail-Risk in Cryptocurrency Markets", *Journal of Empirical
  Finance* 50, 1-19.** Tail dependence within crypto, and the closest existing work to this
  study's question.

## Neighbours on this desk

**238-betting-against-beta**, **419-downside-beta-equities**, **142-bitcoin-correlation**,
**604-crypto-equity-beta**, **988-bitcoin-volatility-decay**, **987-silver-high-beta-gold**.
