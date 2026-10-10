# Sources & literature map — Study 1020 (More Volatile Than Ever?)

## The claim and its mechanisms

- **CFTC & SEC (2010), *Findings Regarding the Market Events of May 6, 2010*** — the joint staff
  report on the flash crash; the episode most often cited for "algorithms make markets more
  volatile".
- **Kirilenko, A., Kyle, A. S., Samadi, M. & Tuzun, T. (2017), "The Flash Crash: High-Frequency
  Trading in an Electronic Market", *Journal of Finance* 72(3), 967-998.** High-frequency
  traders did not trigger the crash but amplified it — the strong form of the claim at
  intraday frequency, which daily data cannot test.
- **Ben-David, I., Franzoni, F. & Moussawi, R. (2018), "Do ETFs Increase Volatility?", *Journal
  of Finance* 73(6), 2471-2535.** Higher ETF ownership raises the volatility of the
  *underlying stocks* — a cross-sectional effect, not a trend in market volatility.

## Has volatility changed over time?

- **Officer, R. R. (1973), "The Variability of the Market Factor of the New York Stock
  Exchange", *Journal of Business* 46(3).** The early documentation that the 1930s
  were exceptional and post-war volatility returned to earlier levels.
- **Schwert, G. W. (1989), "Why Does Stock Market Volatility Change Over Time?", *Journal of
  Finance* 44(5), 1115-1153.** The classic long-sample study: volatility varies a great deal
  and is highest in the Depression, without a secular trend, and it is only loosely tied to
  macro volatility.
- **Campbell, J. Y., Lettau, M., Malkiel, B. G. & Xu, Y. (2001), "Have Individual Stocks Become
  More Volatile? An Empirical Exploration of Idiosyncratic Risk", *Journal of Finance* 56(1),
  1-43.** Firm-level volatility rose from 1962 to 1997 while market volatility showed no
  trend. That is the caveat this study inherits: it measures the market only.

## Measuring volatility

- **Parkinson, M. (1980), "The Extreme Value Method for Estimating the Variance of the Rate of
  Return", *Journal of Business* 53(1), 61-65.** The range estimator used on the arch OHLC tape.
- **Garman, M. B. & Klass, M. J. (1980), "On the Estimation of Security Price Volatilities from
  Historical Data", *Journal of Business* 53(1), 67-78.**
- **Andersen, T. G., Bollerslev, T., Diebold, F. X. & Labys, P. (2003), "Modeling and
  Forecasting Realized Volatility", *Econometrica* 71(2), 579-625.** Realised volatility as
  the object to forecast, and its strong persistence.
- **Mandelbrot, B. (1963), "The Variation of Certain Speculative Prices", *Journal of Business*
  36(4), 394-419.** The fat tails and volatility clustering behind the extreme-day counts.

## Trend inference under persistence

- **Newey, W. K. & West, K. D. (1987), "A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix", *Econometrica* 55(3), 703-708.**
- **Vogelsang, T. J. (1998), "Trend Function Hypothesis Testing in the Presence of Serial
  Correlation", *Econometrica* 66(1), 123-148.** The t-PS family of trend tests that stay
  valid even near a unit root. It is named in the brief and left as a fork; we use the
  related fixed-b test below.
- **Kiefer, N. M., Vogelsang, T. J. & Bunzel, H. (2000), "Simple Robust Testing of Regression
  Hypotheses", *Econometrica* 68(3), 695-714.** The KVB statistic used here. We simulate its
  critical value (5.91 for a 5% two-sided trend test) instead of taking it from a table.
- **Kiefer, N. M. & Vogelsang, T. J. (2002), "Heteroskedasticity-Autocorrelation Robust Standard
  Errors Using the Bartlett Kernel without Truncation", *Econometrica* 70(5), 2093-2095.**
- **Künsch, H. R. (1989), "The Jackknife and the Bootstrap for General Stationary
  Observations", *Annals of Statistics* 17(3), 1217-1241.** The block bootstrap.
- **Politis, D. N. & Romano, J. P. (1994), "The Stationary Bootstrap", *Journal of the American
  Statistical Association* 89(428), 1303-1313.**

## Counting and forecasting

- **Garwood, F. (1936), "Fiducial Limits for the Poisson Distribution", *Biometrika* 28.** The exact Poisson intervals on the extreme-day rates.
- **Diebold, F. X. & Mariano, R. S. (1995), "Comparing Predictive Accuracy", *Journal of
  Business & Economic Statistics* 13(3), 253-263.**
- **Patton, A. J. (2011), "Volatility Forecast Comparison Using Imperfect Volatility Proxies",
  *Journal of Econometrics* 160(1), 246-256.** Why QLIKE is the loss function: it ranks
  forecasts consistently when the realised-volatility proxy is noisy.

## Perception

- **Tversky, A. & Kahneman, D. (1973), "Availability: A Heuristic for Judging Frequency and
  Probability", *Cognitive Psychology* 5(2), 207-232.** We judge how frequent something is by
  how easily examples come to mind. The decade you remember serves as the reference class.

## Data provenance

| Tape | Source | Pin | Notes |
|---|---|---|---|
| `frenchdata` | Kenneth R. French Data Library, monthly Mkt-RF and RF, frozen inside the `arch` package | SHA-256 `f23436727b01d879…` | `Mkt-RF + RF` = **total** return of the US value-weighted market. 1926-07 → 2018-11; annual statistics use 1927-2017. |
| `sp500_index` | S&P 500 daily index level, frozen inside the `skfolio` 1.8.1 wheel (fetched once by `quantlab.bundled.fetch_skfolio`, cached) | SHA-256 `894c34431c284f86…` | **Price** index, so no dividends are included. 1990-01-02 → 2022-12-28, and the partial year 2022 is dropped. |
| `sp500` | S&P 500 daily OHLC, frozen inside the `arch` package | SHA-256 `1e028cbb9c400cc0…` | **Price** index. 1999-01-04 → 2018-12-31. The Parkinson estimator uses High/Low only. |

All three files are read through [`quantlab/bundled.py`](../../../quantlab/bundled.py), which
refuses any file whose bytes differ from the pin. The content fingerprints for each run are
printed in [results.md](results.md).

## Neighbours on this desk

- **817-realized-volatility-trend** trades the cross-section of stock-level volatility *changes*
  (a Mirage there). This study tests the market's volatility *level* over time.
- **992-vol-clustering-halflife** asks how long a volatility storm lasts. Its persistence is the
  reason the trend tests here need to be robust.
- **988-bitcoin-volatility-decay** asks the mirror-image question ("volatility is falling") of
  Bitcoin, using the same start-date trap.
- **1002-best-days-missed** treats extreme days as a story about returns, not about how often they happen.
- **965-range-vol-estimators** compares the range estimators used as the second estimator here.
