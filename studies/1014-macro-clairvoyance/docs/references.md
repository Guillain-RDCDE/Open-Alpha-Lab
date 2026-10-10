# Sources & literature map — Study 1014 (The Perfect Macro Forecaster)

## Stock returns and real activity — who leads whom

- **Fama, E. F. (1981), "Stock Returns, Real Activity, Inflation, and Money", *American Economic
  Review* 71(4), 545-565.** Stock returns are tied to *future* real activity; the negative
  stock-inflation relation proxies for that link. The premise this study grants at full strength.
- **Fama, E. F. (1990), "Stock Returns, Expected Returns, and Real Activity", *Journal of
  Finance* 45(4), 1089-1108.** Quarterly and annual returns are strongly correlated with
  *future* production growth — the lead in this study's §1.
- **Schwert, G. W. (1990), "Stock Returns and Real Activity: A Century of Evidence", *Journal of
  Finance* 45(4), 1237-1257.** The same lead over 1889-1988.
- **Chen, N.-F., Roll, R. & Ross, S. A. (1986), "Economic Forces and the Stock Market",
  *Journal of Business* 59(3), 383-403.** Macro variables as priced risk factors — the academic
  root of "know the economy, know the market".
- **Stock, J. H. & Watson, M. W. (2003), "Forecasting Output and Inflation: The Role of Asset
  Prices", *Journal of Economic Literature* 41(3), 788-829.** The reverse direction — asset
  prices as (unstable) predictors of the economy — surveyed.

## The value of market timing and perfect foresight

- **Sharpe, W. F. (1975), "Likely Gains from Market Timing", *Financial Analysts Journal*.**
  How accurate a timer must be to beat buy-and-hold; the ancestor of §5's ceiling ladder.
- **Henriksson, R. D. & Merton, R. C. (1981), "On Market Timing and Investment Performance. II.
  Statistical Procedures for Evaluating Forecasting Skills", *Journal of Business* 54(4),
  513-533.** Hit rates on the market's direction as the measure of timing skill.

## Real-time data and revisions

- **Croushore, D. & Stark, T. (2001), "A Real-Time Data Set for Macroeconomists", *Journal of
  Econometrics* 105(1), 111-130.** Why a final-vintage macro series is not what anybody knew at
  the time — the reason this study's Tradability stamp is capped at Fragile.

## Method

- **Newey, W. K. & West, K. D. (1987), "A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix", *Econometrica* 55(3), 703-708.**
- **Politis, D. N. & Romano, J. P. (1994), "The Stationary Bootstrap", *Journal of the American
  Statistical Association* 89(428), 1303-1313.** Block resampling for dependent data.
- **Ledoit, O. & Wolf, M. (2008), "Robust Performance Hypothesis Testing with the Sharpe Ratio",
  *Journal of Empirical Finance* 15(5), 850-859.** Bootstrap inference on a Sharpe *difference*,
  the comparison used throughout.

## Data provenance

- **US macro:** `statsmodels.datasets.macrodata` (statsmodels 0.15.0), quarterly 1959Q1-2009Q3,
  **final vintage** as compiled for the package from US official statistics (BEA real GDP, BLS
  unemployment and CPI, Federal Reserve 3-month T-bill). `macrodata.csv` SHA-256
  `d93c0d3a7a77ef83c3af14e46032bb1d02ae3a512b22ab94159a8ca226fcf708`, checked in the tests.
- **US equity:** Kenneth R. French Data Library, monthly Fama-French factors (`Mkt-RF`, `RF`), as
  shipped in `arch` 8.0.0 (`frenchdata.csv.gz`, SHA-256 pinned in `quantlab/bundled.py`).
  `Mkt-RF + RF` is the CRSP value-weighted **total return**; `RF` is the one-month T-bill.
- Loaded through [`quantlab/bundled.py`](../../../quantlab/bundled.py); fully offline.

## Neighbours on this desk

**877-gdpnow-revisions** (daily nowcast revisions), **387-economic-surprise-index** (data
surprises), **268-sahm-rule** (recession trigger), **626-unemployment-trend-timing**
(unemployment as a veto on trend-following), **602-macro-announcement-premium** and
**644-cpi-day-drift** (returns on release days).
