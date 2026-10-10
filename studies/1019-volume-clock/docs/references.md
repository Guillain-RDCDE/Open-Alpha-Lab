# Sources & literature map — Study 1019 (The Market Runs on Volume Time)

## The claim: subordination and the volume clock

- **Clark, P. K. (1973), "A Subordinated Stochastic Process Model with Finite Variance for
  Speculative Prices", *Econometrica* 41(1), 135-155.** The source. Daily price changes as a
  normal process evaluated at a random "operational time" proxied by trading volume; the fat
  tails of calendar-time returns are the mixture. Section 1 tests his clock directly.
- **Ané, T. & Geman, H. (2000), "Order Flow, Transaction Clock, and Normality of Asset
  Returns", *Journal of Finance* 55(5), 2259-2284.** The strongest modern version: with the
  number of trades as the clock, intraday stock returns are close to normal. The claim as
  steelmanned in the README.
- **Mandelbrot, B. & Taylor, H. M. (1967), "On the Distribution of Stock Price Differences",
  *Operations Research* 15(6), 1057-1062.** The original subordination idea in "transaction
  time", which Clark's finite-variance version answers.

## Volume and volatility

- **Karpoff, J. M. (1987), "The Relation Between Price Changes and Trading Volume: A Survey",
  *Journal of Financial and Quantitative Analysis* 22(1), 109-126.** The stylised fact that
  volume and absolute price changes move together — section 2.
- **Tauchen, G. E. & Pitts, M. (1983), "The Price Variability-Volume Relationship on
  Speculative Markets", *Econometrica* 51(2), 485-505.** The mixture-of-distributions
  hypothesis: volume and volatility driven jointly by the rate of information arrival.
- **Lamoureux, C. G. & Lastrapes, W. D. (1990), "Heteroskedasticity in Stock Return Data:
  Volume versus GARCH Effects", *Journal of Finance* 45(1), 221-229.** Same-day volume in the
  GARCH variance equation absorbs much of the GARCH persistence — the "on top of GARCH" check in
  section 3 of the results, and the reason the same-day result does not carry over to a
  forecast.
- **Brooks, C. (1998), "Predicting Stock Index Volatility: Can Market Volume Help?", *Journal
  of Forecasting*.** The direct antecedent of section 5: lagged index volume adds little
  to out-of-sample volatility forecasts.

## Volatility models, estimators and forecast evaluation

- **Bollerslev, T. (1986), "Generalized Autoregressive Conditional Heteroskedasticity",
  *Journal of Econometrics* 31(3), 307-327.** GARCH(1,1), the usual benchmark.
- **Corsi, F. (2009), "A Simple Approximate Long-Memory Model of Realized Volatility",
  *Journal of Financial Econometrics* 7(2), 174-196.** The HAR model; here in logs, on Parkinson
  range-variance.
- **Parkinson, M. (1980), "The Extreme Value Method for Estimating the Variance of the Rate of
  Return", *Journal of Business* 53(1), 61-65.** The range estimator used as the upper-benchmark
  clock and as HAR's input.
- **Patton, A. J. (2011), "Volatility Forecast Comparison Using Imperfect Volatility Proxies",
  *Journal of Econometrics* 160(1), 246-256.** Why QLIKE (and MSE) remain consistent rankings
  when the target is a noisy proxy such as a squared return.
- **Diebold, F. X. & Mariano, R. S. (1995), "Comparing Predictive Accuracy", *Journal of
  Business & Economic Statistics* 13(3), 253-263.**
- **Newey, W. K. & West, K. D. (1987), "A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix", *Econometrica* 55(3), 703-708.**
- **Jarque, C. M. & Bera, A. K. (1987), "A Test for Normality of Observations and Regression
  Residuals", *International Statistical Review* 55(2), 163-172.**
- **Politis, D. N. & Romano, J. P. (1992), "A Circular Block-Resampling Procedure for Stationary
  Data", in *Exploring the Limits of Bootstrap* (Wiley).** The circular block bootstrap behind
  every interval here.
- **Moreira, A. & Muir, T. (2017), "Volatility-Managed Portfolios", *Journal of Finance* 72(4),
  1611-1644.** The vol-targeting overlay of section 6, and why it is a reasonable "standard"
  use of a variance forecast.

## Data provenance

- **S&P 500 and Nasdaq Composite daily OHLCV, 1999-01-04 → 2018-12-31**, the `sp500` and
  `nasdaq` datasets shipped inside the [`arch`](https://github.com/bashtage/arch) package
  (Kevin Sheppard), originally from Yahoo! Finance. Read through
  [`quantlab/bundled.py`](../../../quantlab/bundled.py), which pins each file by SHA-256 and
  refuses a mismatched tape. **Price indices** (no dividends); `Volume` is the composite volume
  Yahoo reports for the index. Two Nasdaq sessions with zero volume are set to missing.
- **Fama-French monthly RF** (Kenneth R. French Data Library), the `frenchdata` dataset in the
  same package, used as the cash leg; December 2018 carries November's rate forward.

## Neighbours on this desk

- **991-aggregational-gaussianity** — returns become more normal as the *calendar* horizon
  lengthens; this study changes the *clock* instead of the horizon.
- **965-range-vol-estimators** — range estimators as volatility measures; here the range is an
  upper-benchmark clock and HAR's input.
- **966-har-vs-garch** — the HAR-vs-GARCH race; here HAR is the baseline and the question is the
  marginal value of volume.
- **992-vol-clustering-halflife** — the persistence that, on these tapes, explains more of the
  fat tail than the volume clock does.
- **993-leverage-effect-asymmetry** — the signed fork of section 7.
