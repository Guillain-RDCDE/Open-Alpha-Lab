# Sources & literature map — Study 992 (How Long Is a Storm?)

## The clustering itself

- **Mandelbrot, B. (1963), "The Variation of Certain Speculative Prices", *Journal of Business*
  36(4), 394-419.** "Large changes tend to be followed by large changes" — the original
  observation.
- **Engle, R. F. (1982), "Autoregressive Conditional Heteroscedasticity with Estimates of the
  Variance of United Kingdom Inflation", *Econometrica* 50(4), 987-1008**, and **Bollerslev, T.
  (1986), *Journal of Econometrics* 31(3), 307-327.** ARCH and GARCH; the persistence parameter
  α + β whose half-life this study computes.

## Why one number is not enough

- **Ding, Z., Granger, C. W. J. & Engle, R. F. (1993), "A Long Memory Property of Stock Market
  Returns and a New Model", *Journal of Empirical Finance* 1(1), 83-106.** The autocorrelation of
  absolute returns decays hyperbolically, not exponentially — so *no* single half-life describes
  it, which is this study's central point arrived at from the other direction.
- **Engle, R. F. & Lee, G. (1999), "A Permanent and Transitory Component Model of Stock Return
  Volatility", in *Cointegration, Causality, and Forecasting*, OUP.** The component GARCH model:
  exactly the fast-plus-slow decomposition that `two_component_fit` estimates
  non-parametrically.
- **Comte, F. & Renault, E. (1998), "Long Memory in Continuous-Time Stochastic Volatility
  Models", *Mathematical Finance* 8(4), 291-323.** The continuous-time version, in which
  volatility has a continuum of timescales.
- **Corsi, F. (2009), "A Simple Approximate Long-Memory Model of Realized Volatility", *Journal
  of Financial Econometrics* 7(2), 174-196.** The HAR model, which works precisely by mixing
  daily, weekly and monthly components rather than choosing one.

## Measurement issues

- **Andersen, T. G. & Bollerslev, T. (1998), "Answering the Skeptics: Yes, Standard Volatility
  Models Do Provide Accurate Forecasts", *International Economic Review* 39(4), 885-905.** Why
  the volatility proxy matters as much as the model.
- **Forsberg, L. & Ghysels, E. (2007), "Why Do Absolute Returns Predict Volatility So Well?",
  *Journal of Financial Econometrics* 5(1), 31-67.** Absolute returns beat squared returns as a
  proxy — the reason `acf_halflife` defaults to `abs`.
- **J.P. Morgan/Reuters (1996), *RiskMetrics — Technical Document*.** Where λ = 0.94 comes from,
  and the fact that it was chosen rather than estimated.

## Neighbours on this desk

**256-volatility-clustering**, **966-garch-vs-har**, **371-vix-term-structure**,
**988-bitcoin-volatility-decay**, **991-aggregational-gaussianity**.
