# Sources & literature map — Study 966 (Forecasting Tomorrow's Vol)

## The fact the whole field rests on

- **Mandelbrot, B. (1963), "The Variation of Certain Speculative Prices", *Journal of Business*
  36(4), 394-419.** "Large changes tend to be followed by large changes" — volatility
  clustering, first stated.
- **Engle, R. F. (1982), "Autoregressive Conditional Heteroscedasticity with Estimates of the
  Variance of United Kingdom Inflation", *Econometrica* 50(4), 987-1007.** ARCH; the Nobel.
- **Bollerslev, T. (1986), "Generalized Autoregressive Conditional Heteroskedasticity",
  *Journal of Econometrics* 31(3), 307-327.** GARCH(1,1), the model this study fits.

## The competitors

- **J.P. Morgan/Reuters (1996), *RiskMetrics Technical Document*, 4th ed.** The EWMA with
  lambda = 0.94 for daily data — one parameter, never estimated, and the baseline that
  embarrasses a lot of fitted models.
- **Corsi, F. (2009), "A Simple Approximate Long-Memory Model of Realized Volatility",
  *Journal of Financial Econometrics* 7(2), 174-196.** HAR-RV: daily, weekly and monthly
  components as a cascade. Three coefficients and an OLS.
- **Hansen, P. R. & Lunde, A. (2005), "A Forecast Comparison of Volatility Models: Does
  Anything Beat a GARCH(1,1)?", *Journal of Applied Econometrics* 20(7), 873-889.** 330 models
  on two series; the honest answer is "not reliably". The intellectual ancestor of this study.
- **Bollerslev, T., Wooldridge, J. M. (1992), "Quasi-Maximum Likelihood Estimation and
  Inference in Dynamic Models with Time-Varying Covariances", *Econometric Reviews* 11(2),
  143-172.** Why fitting a Gaussian likelihood to fat-tailed returns is still consistent —
  the licence for the QML fit used here.

## Scoring, and the proxy problem

- **Andersen, T. G. & Bollerslev, T. (1998), "Answering the Skeptics: Yes, Standard Volatility
  Models Do Provide Accurate Forecasts", *International Economic Review* 39(4), 885-905.** The
  paper that showed the old "GARCH has an R² of 5%" complaint was an artefact of using squared
  daily returns as the target.
- **Patton, A. J. (2011), "Volatility Forecast Comparison Using Imperfect Volatility Proxies",
  *Journal of Econometrics* 160(1), 246-256.** MSE and QLIKE are the robust loss functions;
  most others rank models incorrectly when the target is noisy.
- **Diebold, F. X. & Mariano, R. S. (1995), *JBES* 13(3), 253-263.** The comparison test.
- **Hansen, P. R. (2005), "A Test for Superior Predictive Ability", *JBES* 23(4), 365-380.**
  What to do when the number of competing models is large — the correction this study does not
  need with four models, and would need with forty.

## Neighbours on this desk

**965-range-vol-estimators** (measurement rather than forecasting),
**817-realized-volatility-trend**, **992-vol-clustering-halflife**, **374-vol-of-vol**,
**130-vol-risk-premium**, **898-managed-vol-equity**, **591-vol-managed-portfolio**.
