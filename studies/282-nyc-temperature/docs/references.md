# References & literature map -- Study 282 (NYC-Temperature)

## The claim under test

The folk claim is that the **temperature on Wall Street** -- the weather at the
New York Stock Exchange -- moves that day's stock returns through trader mood:
cold, gloomy days breed risk aversion (selling), mild days breed optimism
(buying). The respectable academic ancestors are about *sunshine/cloud cover*,
not temperature per se, but the temperature version is the popular simplification.

- **Saunders, E. M. (1993).** "Stock Prices and Wall Street Weather." *American
  Economic Review*, 83(5), 1337-1345. The original: NYC cloud cover is weakly
  negatively related to NYSE returns. Small effect, much-cited, much-criticized.

- **Hirshleifer, D. & Shumway, T. (2003).** "Good Day Sunshine: Stock Returns
  and the Weather." *Journal of Finance*, 58(3), 1009-1032. Finds a sunshine
  effect across 26 stock-exchange cities, but notes it is not exploitable after
  transaction costs and is fragile to specification.

- **Kamstra, M., Kramer, L. & Levi, M. (2003).** "Winter Blues: A SAD Stock
  Market Cycle." *American Economic Review*, 93(1), 324-343. Seasonal Affective
  Disorder and returns -- a seasonal (not daily-weather) mood channel.

## Why temperature is the weak version of the story

- **The mood mechanism is plausible but tiny.** Laboratory mood effects on
  risk-taking are fractions of a standard deviation. Translated to an index, the
  implied daily tilt is a basis point or two -- against ~100 bps/day of return
  volatility. The signal-to-noise ratio is hopeless for daily data.

- **Temperature is the wrong proxy.** Saunders and Hirshleifer-Shumway used
  *cloud cover / sunshine*, the variable most tied to mood. Temperature is only
  loosely correlated with sunshine and adds the confound of the seasonal cycle.
  We remove the seasonal cycle by working with the **anomaly** (temperature minus
  its day-of-year normal), so calendar seasonality (e.g. the January effect)
  cannot masquerade as a weather signal.

- **One station is not the marginal trader.** We use a single Central Park
  record. The marginal price-setter is geographically dispersed (and increasingly
  an algorithm with no mood at all). A single-station proxy is noisy; if anything
  that noise biases *toward* a spurious finding in a data-mined sort.

## The autocorrelation trap and the correct inference

- **Weather is highly autocorrelated.** A cold *spell* lasts a week; those days
  are not independent observations. A naive t-test over ~16,000 daily
  observations badly over-states significance. We use a **Newey-West HAC**
  standard error (lag length `floor(4*(n/100)^(2/9))`) on every mean and on the
  OLS slope so the t-stat reflects the true number of independent episodes.

- **Permutation test.** We shuffle the temperature labels 2,000 times and rebuild
  the cold-minus-warm spread to map the null distribution non-parametrically --
  robust to the non-normal, fat-tailed daily return distribution.

- **The bar for REAL.** A robust HAC `|t| >= 2` on the real tape. Literature
  support alone (a published sunshine effect) is at most WEAK; we require the
  signal to clear the noise floor here, and it does not.

## Method lineage

- **Tercile sort.** Bottom vs top temperature-anomaly tercile; spread = mean
  cold-day return minus mean warm-day return.
- **OLS sensitivity slope.** Daily return (bps) on the standardized anomaly, with
  a HAC-robust slope SE -- the regression analogue of the sort.
- **Newey-West HAC.** `floor(4*(n/100)^(2/9))` lags; the standard correction for
  serially correlated regressors and residuals.
- **Net-of-cost backtest.** One-day execution lag (yesterday's anomaly -> today's
  position), 1 bp one-way cost x NAV on every position change (shorts pay the
  same one-way cost; borrow folded in); gross vs net Sharpe and turnover.

## Data sources

- **^GSPC daily.** Yahoo! Finance S&P 500 daily *price* index (price-only, not
  total-return). Cache-only via yfinance at `_cache/gspc_daily.parquet`; the
  network is touched only on an explicit `fetch=True`.
- **NYC temperature.** Curated monthly anomaly table hardcoded in `data.py`,
  representative of the Central Park (NOAA/NCEI) record and GISS NYC gridcell,
  used to drive a deterministic daily temperature builder.

## Related desk studies

- **[Study 158 -- Super-Bowl](../../158-super-bowl/)**: the same teardown
  template (hardcoded event/series table + honest null + positive control) for a
  spurious folklore predictor.
- The SAD / daylight-saving / seasonal-mood teardowns elsewhere on the desk share
  the "plausible mood story, microscopic effect, giant noise floor" structure.
