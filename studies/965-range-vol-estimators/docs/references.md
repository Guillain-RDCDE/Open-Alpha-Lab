# Sources & literature map — Study 965 (The Range Estimators)

## The estimators themselves

- **Parkinson, M. (1980), "The Extreme Value Method for Estimating the Variance of the Rate
  of Return", *Journal of Business* 53(1), 61-65.** The original: `ln(H/L)^2 / (4 ln 2)`,
  with the famous ~5x efficiency gain over squared close-to-close returns under a driftless
  geometric Brownian motion **with no gaps**.
- **Garman, M. B. & Klass, M. J. (1980), "On the Estimation of Security Price Volatilities
  from Historical Data", *Journal of Business* 53(1), 67-78.** The minimum-variance
  combination of the range and the open-to-close move; quoted at ~7-8x efficiency.
- **Rogers, L. C. G. & Satchell, S. E. (1991), "Estimating Variance from High, Low and Closing
  Prices", *Annals of Applied Probability* 1(4), 504-512.** Drift-robust: the estimator stays
  unbiased when the day has a trend, which Parkinson and Garman-Klass do not.
- **Yang, D. & Zhang, Q. (2000), "Drift-Independent Volatility Estimation Based on High, Low,
  Open, and Close Prices", *Journal of Business* 73(3), 477-491.** The one that handles the
  **overnight gap** — the reason it is this study's honest default.
- **Garman & Klass (1980), section on opening jumps**, and **Beckers, S. (1983), "Variance of
  Security Price Returns Based on High, Low, and Closing Prices", *Journal of Business*
  56(1), 97-112.** Early empirical work already noting that the theoretical gains shrink on
  real data.

## Why the gains shrink in practice

- **Marsh, T. A. & Rosenfeld, E. R. (1986), "Non-Trading, Market Making, and Estimates of
  Stock Price Volatility", *Journal of Financial Economics* 15(3), 359-372.** Discrete
  trading biases the observed range downward: the true high and low are never printed.
- **Andersen, T. G. & Bollerslev, T. (1998), "Answering the Skeptics: Yes, Standard Volatility
  Models Do Provide Accurate Forecasts", *International Economic Review* 39(4), 885-905.**
  The realised-variance benchmark that made honest volatility forecast evaluation possible —
  and the benchmark this study cannot build without intraday data.
- **Alizadeh, S., Brandt, M. W. & Diebold, F. X. (2002), "Range-Based Estimation of Stochastic
  Volatility Models", *Journal of Finance* 57(3), 1047-1091.** The modern case *for* the
  range: log-range is nearly Gaussian and robust to microstructure noise.
- **Shu, J. & Zhang, J. E. (2006), "Testing Range Estimators of Historical Volatility",
  *Journal of Futures Markets* 26(3), 297-313.** Empirical horse race with the same finding
  this study reproduces: the ranking survives, the magnitudes do not.

## Scoring a volatility forecast

- **Patton, A. J. (2011), "Volatility Forecast Comparison Using Imperfect Volatility Proxies",
  *Journal of Econometrics* 160(1), 246-256.** Why MSE and QLIKE are the two loss functions
  that stay honest when the target is a noisy proxy — the reason both are reported here.
- **Diebold, F. X. & Mariano, R. S. (1995), "Comparing Predictive Accuracy", *Journal of
  Business & Economic Statistics* 13(3), 253-263.** The test used to compare loss series.
- **Hansen, P. R. & Lunde, A. (2005), "A Forecast Comparison of Volatility Models: Does
  Anything Beat a GARCH(1,1)?", *Journal of Applied Econometrics* 20(7), 873-889.** The
  companion warning about drawing conclusions from a horse race with many entrants.

## Neighbours on this desk

**966-har-vs-garch** (forecasting models on the same tapes), **817-realized-volatility-trend**,
**992-vol-clustering-halflife**, **374-vol-of-vol**, **130-vol-risk-premium**,
**788-overnight-intraday-tug-of-war**, and **812-corwin-schultz-spread** (the other classic
high-low estimator, for the spread rather than the variance).
