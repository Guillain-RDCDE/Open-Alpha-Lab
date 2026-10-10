# Sources & literature map — Study 1023 (The Cushing Glut)

## The Brent–WTI spread and the Cushing episode

- **Fattouh, B. (2010), "The dynamics of crude oil price differentials", *Energy Economics*
  32(2).** Differentials between benchmark crudes treated as tethered by arbitrage; tests
  for stationarity and threshold adjustment. This is the academic form of the claim.
- **Büyükşahin, B., Lee, T. K., Moser, J. T. & Robe, M. A. (2013), "Physical Markets, Paper
  Markets and the WTI-Brent Spread", *The Energy Journal* 34(3).** Ties the post-2010 widening
  to physical constraints at Cushing, with financial-market factors in a supporting role.
- **Borenstein, S. & Kellogg, R. (2014), "The Incidence of an Oil Glut: Who Benefits from Cheap
  Crude Oil in the Midwest?", *The Energy Journal* 35(1).** The glut seen from the downstream
  side, as a transport bottleneck pricing the landlocked barrel down.

## Unit roots, stationarity and cointegration

- **Dickey, D. A. & Fuller, W. A. (1979), "Distribution of the Estimators for Autoregressive
  Time Series with a Unit Root", *Journal of the American Statistical Association* 74(366),
  427-431.** The ADF test's ancestor.
- **Kwiatkowski, D., Phillips, P. C. B., Schmidt, P. & Shin, Y. (1992), "Testing the Null
  Hypothesis of Stationarity against the Alternative of a Unit Root", *Journal of Econometrics*
  54, 159-178.** KPSS. We run it next to ADF because the two nulls are opposite.
- **Engle, R. F. & Granger, C. W. J. (1987), "Co-integration and Error Correction:
  Representation, Estimation, and Testing", *Econometrica* 55(2), 251-276.**
- **MacKinnon, J. G. (1996), "Numerical Distribution Functions for Unit Root and Cointegration
  Tests", *Journal of Applied Econometrics* 11(6), 601-618.** The p-values statsmodels reports.
- **Perron, P. (1989), "The Great Crash, the Oil Price Shock, and the Unit Root Hypothesis",
  *Econometrica* 57(6), 1361-1401.** A level break makes a stationary series look like a unit
  root to ADF. §4 of the results reproduces this on synthetic data at the real tape's break size.

## Structural breaks

- **Andrews, D. W. K. (1993), "Tests for Parameter Instability and Structural Change with
  Unknown Change Point", *Econometrica* 61(4), 821-856.** The sup-F test and its 15%-trimming
  critical values. We show those values and then replace them with a bootstrap, because they
  assume weak dependence.
- **Bai, J. & Perron, P. (1998), "Estimating and Testing Linear Models with Multiple
  Structural Changes", *Econometrica* 66(1), 47-78.**
- **Bai, J. & Perron, P. (2003), "Computation and Analysis of Multiple Structural Change
  Models", *Journal of Applied Econometrics* 18(1), 1-22.** The dynamic program used in
  `bai_perron`, and the recommendation of the LWZ criterion.
- **Liu, J., Wu, S. & Zidek, J. V. (1997), "On Segmented Multivariate Regression", *Statistica
  Sinica* 7.** The LWZ information criterion.
- **Ploberger, W. & Krämer, W. (1992), "The CUSUM Test with OLS Residuals", *Econometrica*
  60(2), 271-285.**

## Mean reversion, half-lives and the averaging artefact

- **Uhlenbeck, G. E. & Ornstein, L. S. (1930), "On the Theory of the Brownian Motion",
  *Physical Review* 36, 823-841.** The OU process behind the synthetic generator.
- **Kendall, M. G. (1954), "Note on Bias in the Estimation of Autocorrelation", *Biometrika*
  41.** OLS AR(1) coefficients are biased down in short samples, so the half-lives here lean fast.
- **Working, H. (1960), "Note on the Correlation of First Differences of Averages in a Random
  Chain", *Econometrica* 28(4), 916-918.** Time-averaged prices acquire spurious autocorrelation
  in their changes. That matters because this tape is monthly averages.
- **Gatev, E., Goetzmann, W. N. & Rouwenhorst, K. G. (2006), "Pairs Trading: Performance of a
  Relative-Value Arbitrage Rule", *Review of Financial Studies* 19(3), 797-827.** The canonical
  spread-fading rule. Our ±2σ entry follows its spirit.

## Inference

- **Newey, W. K. & West, K. D. (1987), "A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix", *Econometrica* 55(3), 703-708.** HAC *t* of
  the monthly P&L (`quantlab.analytics.mean_tstat_hac`) and the CUSUM scaling.
- **Politis, D. N. & Romano, J. P. (1994), "The Stationary Bootstrap", *Journal of the American
  Statistical Association* 89(428), 1303-1313.** The block-bootstrap family behind the Sharpe
  intervals (`quantlab.stats.sharpe_ci_bootstrap`, circular blocks of 12 months).

## Data provenance

- **Tape:** `arch/data/crude/crude.csv.gz` inside the `arch` package, read through
  [`quantlab/bundled.py`](../../../quantlab/bundled.py) with SHA-256 pin
  `4190b34f6130069a5b91ec03e8ea1a4dfffba8d379033f9cf3934fa695da7ef7`. Monthly Brent and WTI
  spot, US$/bbl, nominal, 1987-05 → 2020-01. The data are FRED-sourced. The `crude` loader's
  own docstring in `arch` is a copy of the Core-CPI one, so the series' construction is
  **verified empirically** rather than taken from the label: `cushing.data.check_monthly_average`
  matches the monthly WTI column to the calendar-month mean of `arch`'s daily FRED WTI tape
  (`DCOILWTICO`) within 1¢ in every one of 380 overlapping full months. Brent is assumed to
  follow the same construction because no daily Brent tape is bundled.
- **Not used:** futures prices, roll yields, Cushing inventories. All three are needed for an
  investable version, and their absence is why every P&L in this study is an upper bound.

## Neighbours on this desk

**05-twin-spread** (GGR pairs on equities), **23-broken-tether** (cointegrated ETF pairs out of
sample), **306-crack-spread** (a refining spread timing refiner equities), **305-gold-oil-ratio**,
**226-crude-seasonality**, **650-heating-oil-seasonality**.
