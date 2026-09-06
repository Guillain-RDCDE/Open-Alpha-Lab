# Sources & literature map — Study 973 (Different Holidays)

## The classic corrections

- **Scholes, M. & Williams, J. (1977), "Estimating Betas from Nonsynchronous Data", *Journal of
  Financial Economics* 5(3), 309-327.** The original statement of the problem and the
  consistent estimator implemented here.
- **Dimson, E. (1979), "Risk Measurement When Shares Are Subject to Infrequent Trading",
  *Journal of Financial Economics* 7(2), 197-226.** The aggregated-coefficients method: regress
  on leads and lags and sum. Simpler than Scholes-Williams and usually indistinguishable from it.
- **Cohen, K. J., Hawawini, G. A., Maier, S. F., Schwartz, R. A. & Whitcomb, D. K. (1983),
  "Friction in the Trading Process and the Estimation of Systematic Risk", *Journal of
  Financial Economics* 12(2), 263-278.** Why the bias grows as the measurement interval
  shrinks — the theoretical basis for the frequency-aggregation fix.
- **Lo, A. W. & MacKinlay, A. C. (1990), "An Econometric Analysis of Nonsynchronous Trading",
  *Journal of Econometrics* 45(1-2), 181-211.** The full model, including the spurious
  autocorrelation that non-synchronous trading induces in portfolio returns.

## International markets specifically

- **Eun, C. S. & Shim, S. (1989), "International Transmission of Stock Market Movements",
  *Journal of Financial and Quantitative Analysis* 24(2), 241-256.** US innovations show up in
  other markets the *next* day — the lead-lag profile this study plots.
- **Hamao, Y., Masulis, R. W. & Ng, V. (1990), "Correlations in Price Changes and Volatility
  Across International Stock Markets", *Review of Financial Studies* 3(2), 281-307.** The
  Tokyo-New York case in particular.
- **Martens, M. & Poon, S.-H. (2001), "Returns Synchronization and Daily Correlation Dynamics
  Between International Stock Markets", *Journal of Banking & Finance* 25(10), 1805-1827.**
  Shows how much of the measured cross-market correlation is a synchronisation artefact — the
  direct ancestor of this study.
- **Burns, P., Engle, R. & Mezrich, J. (1998), "Correlations and Volatilities of Asynchronous
  Data", *Journal of Derivatives* 5(4), 7-18.** A practical adjustment for portfolio work,
  which is where the consequence in section 4 lands.

## Why it matters downstream

- **Michaud, R. O. (1989), "The Markowitz Optimization Enigma", *Financial Analysts Journal*
  45(1), 31-42.** An optimiser fed a downward-biased correlation matrix believes diversification
  is cheaper than it is — the promised-versus-delivered gap measured here.
- **Getmansky, M., Lo, A. W. & Makarov, I. (2004), *Journal of Financial Economics* 74(3),
  529-609.** The same statistical problem in its most extreme form (illiquid hedge-fund
  returns), with the unsmoothing machinery this study's fixes are a light version of.

## Neighbours on this desk

**917-nav-staleness-timezone**, **578-cross-asset-correlation-regime**, **634-us-leads-the-world**,
**146-country-momentum**, **613-currency-hedged-etf-carry**, **975-covariance-shrinkage**,
**970-sqrt-time-scaling**.
