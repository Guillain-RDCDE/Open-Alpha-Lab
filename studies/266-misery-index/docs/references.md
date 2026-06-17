# References & literature map — Study 266 (Misery-Index)

## The claim under test

The **misery index** is the sum of the headline inflation rate and the
unemployment rate, popularized in the 1970s and attributed to economist
**Arthur Okun**. It is a *welfare* indicator: a higher number means more
households are hurting from either rising prices or lost jobs. The folk-finance
leap is to treat it as a *market* indicator. Two opposite stories circulate:

- **Leading-bear.** High misery signals a deteriorating economy, so equities
  should underperform in the year ahead (a negative misery→return slope).
- **Contrarian / maximum-pessimism.** Misery peaks at the bottom of the cycle,
  precisely when forward returns are highest — "be greedy when others are
  fearful" (a positive misery→return slope). This is the Templeton/Buffett
  reading.

We test both against the next-calendar-year S&P 500 return.

## Why neither story survives

- **Tiny, serially-correlated n.** There are ~76 annual observations
  (1949–2025), but inflation and unemployment move in multi-year regimes, so the
  *effective* sample is far smaller. Ordinary OLS t-stats are badly overstated;
  the honest statistic is a Newey-West (HAC) t. With ~17% annual equity vol and
  76 obs, the minimum detectable per-1-SD slope at |t| = 2 is on the order of
  3–4%/yr — larger than anything in the data.

- **The base-rate trap.** The S&P rises in ~74% of calendar years
  unconditionally. Any "buy when misery is high" rule inherits this up-rate for
  free; the question is whether high-misery years beat the unconditional rate,
  not 50%.

- **Confounding with valuation and rates.** Insofar as any "buy the bottom"
  signal works, it is better captured by valuation (CAPE) or the equity risk
  premium than by the misery index, which is a coincident macro state, not a
  forward-looking price.

## Academic and practitioner literature

- **Okun, A. M. (1970s).** The misery (or "discomfort") index — inflation plus
  unemployment — entered policy debate as a welfare gauge, not a market timer.

- **Fama, E. F. (1981).** "Stock Returns, Real Activity, Inflation, and Money."
  *American Economic Review*, 71(4), 545–565. The classic result that the
  *negative* stock–inflation correlation is a proxy effect (inflation forecasts
  lower real activity), not a structural risk premium — cautioning against
  reading a clean macro→return channel into inflation.

- **Campbell, J. Y. & Vuolteenaho, T. (2004).** "Inflation Illusion and Stock
  Prices." *American Economic Review*, 94(2), 19–23. Documents that investors
  misprice equities relative to inflation — a behavioral confound for any
  misery→return regression.

- **Welch, I. & Goyal, A. (2008).** "A Comprehensive Look at the Empirical
  Performance of Equity Premium Prediction." *Review of Financial Studies*,
  21(4), 1455–1508. The canonical demonstration that most macro predictors of
  the equity premium (including inflation and labour-market variables) fail
  *out-of-sample* despite in-sample significance — the central caution for this
  study.

- **Cochrane, J. H. (2008).** "The Dog That Did Not Bark: A Defense of Return
  Predictability." *Review of Financial Studies*, 21(4), 1533–1575. The
  counterpoint on predictability and why HAC inference matters with persistent
  regressors.

## Method lineage

- **Newey-West (HAC) standard errors.** Newey, W. K. & West, K. D. (1987).
  "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation
  Consistent Covariance Matrix." *Econometrica*, 55(3), 703–708. We compute the
  Bartlett-kernel HAC covariance by hand (pure numpy) with a 3-lag bandwidth;
  the slope's HAC t-stat is the headline statistic. REAL requires |t_HAC| ≥ 2.

- **Predictive regression.** Forward annual return on the standardized misery
  level (and, separately, its YoY change). Slopes are reported per 1-SD of the
  predictor.

- **Tercile sort + Welch t-test.** Split years into low/mid/high-misery
  terciles; compare top vs bottom forward-return means with an unequal-variance
  t-test (`scipy.stats.ttest_ind(equal_var=False)`).

- **Positive control.** A synthetic (misery, forward-return) generator with a
  planted per-1-SD beta confirms the engine recovers a real effect (|t_HAC| ≥ 2
  at beta ≈ 0.02 on 400 obs) and is quiet under the null (beta = 0).

## Data sources

- **CPI-U, December year-on-year inflation.** U.S. Bureau of Labor Statistics,
  series CUUR0000SA0 (mirrored on FRED as CPIAUCNS, 12-month % change).
  Hardcoded in `data.py`, 1948–2025.
- **Unemployment rate, December, seasonally adjusted.** U.S. Bureau of Labor
  Statistics, series LNS14000000 (FRED: UNRATE). Hardcoded in `data.py`.
- **S&P 500 calendar-year price returns.** December/December closes from the
  repo-level cache `_cache/^GSPC_split_only.parquet` (Yahoo! ^GSPC, price-only,
  no dividends). Cache-only by default; `fetch=True` triggers a one-time
  `yfinance` download.

## Related desk studies

- **[Study 120 — Excess-CAPE-Yield](../../120-excess-cape-yield/)**: the valuation
  channel that *does* carry some forward content, against which the misery index
  is a weak macro proxy.
- **[Study 158 — Super-Bowl](../../158-super-bowl/)**: the base-rate-trap teardown
  template — same honest-null discipline applied to a folklore predictor.
- **[Study 223 — Same-Month-Seasonality](../../223-same-month-seasonality/)**: the
  synthetic-panel positive-control pattern this study mirrors.
