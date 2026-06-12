# References & literature map — Study 84 (Moon-Math)

## The claim under test

- **PlanB (2019).** *"Modeling Bitcoin Value with Scarcity"* — published on Medium and
  later formalised in a white-paper circulated on Twitter/X (@100trillionUSD). The core
  assertion: `log(BTC price) = -1.84 + 3.3 * log(S2F)` where S2F = cumulative supply
  divided by annual new issuance. The in-sample R² was reported at ~0.94, the model
  was used to forecast a $100k+ BTC price by end-2021. The paper has no peer review;
  the data and methodology are publicly available and widely reproduced.

- **The S2F model in practice.** The forecast for end-2021 was ~$100–288k depending
  on the variant. BTC peaked at approximately $57–69k in November 2021 then collapsed
  to $15,787 by November 2022. The model's defenders invoked "phase transitions" and
  model uncertainty; critics pointed to the OOS failure as disconfirmatory evidence.

## The spurious regression literature — why the headline R² is uninformative

- **Granger & Newbold (1974).** *"Spurious Regressions in Econometrics"* (Journal of
  Econometrics). The seminal paper: two independent random walks regressed on each other
  produce a high t-statistic and R² purely by coincidence of trending. The S2F and BTC
  price are both I(1) (non-stationary, trending) processes — the classical spurious
  regression scenario.

- **Phillips (1986).** *"Understanding Spurious Regressions in Econometrics"* (Journal of
  Econometrics). Formalises the asymptotic theory: as sample size grows, t-stats and R²
  diverge to infinity in a spurious regression, giving the false impression of a
  tightening relationship. Our result (time R² > S2F R²) is consistent with both being
  proxies for the same upward march of time.

- **Engle & Granger (1987).** *"Co-integration and Error Correction: Representation,
  Estimation, and Testing"* (Econometrica). The remedy: if two I(1) series are genuinely
  cointegrated, their OLS residuals will be *stationary* (I(0)); a failing ADF test on
  residuals (p > 0.05) is evidence of spuriousness. On the full 2014–2026 tape, the
  log(time) residuals fail this test while S2F residuals (on the full sample) pass — but
  the first-diff evidence and the OOS collapse still rule out a stable structural model.

- **Yule (1926).** *"Why Do We Sometimes Get Nonsense Correlations Between Time-Series?"*
  (Journal of the Royal Statistical Society). The original 'nonsense correlation' paper —
  a R² of 0.95 between two completely unrelated trending series was a historical example
  cited to caution against exactly this type of inference.

## Why the log(time) alternative is decisive

- **The time-as-regressor argument.** If `log(S2F)` and `log(time since genesis)` both
  fit `log(BTC price)` equally well (and log(time) fits *better* in our study, with
  R² = 0.90 vs 0.69 in-sample), then the scarcity narrative is redundant. Any variable
  that monotonically increases — number of internet users, cumulative bitcoin transactions,
  number of blockchain forks — would produce a similar R² on a log scale. This does not
  mean scarcity *doesn't matter*; it means the regression cannot identify its effect
  separately from any other trend.

- **Coin & Jorda (various).** The general problem of "trend-on-trend" regressions in
  financial data is well-documented; see also Valkanov (2003), *"Long-horizon Regressions:
  Theoretical Results and Applications"* (Journal of Financial Economics), for the
  inferential problems introduced by overlapping and trending regressors.

## The first-difference test — forecasting content

- **Pagan & Wickens (1989).** *"A Survey of Some Recent Econometric Methods"* (Economic
  Journal). The standard recommendation: if you believe you have a levels relationship,
  you should find a corresponding first-difference relationship. An R² near zero in
  first differences, when levels R² is 0.80, is disconfirmatory.

- **Ferson, Sarkissian & Simin (2003).** *"Spurious Regressions in Financial Economics?"*
  (Journal of Finance). Documents that persistent predictors (exactly as S2F is, since it
  is nearly constant within each epoch) produce spuriously high t-statistics in predictive
  regressions, even after controlling for persistence.

## Method lineage (the desk's shared engine)

- **Newey & West (1987).** *"A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix"* (Econometrica) — the HAC standard errors
  used in the first-difference t-stat (`strategy.first_diff_regression`).

- **Augmented Dickey-Fuller test.** Dickey & Fuller (1979), *"Distribution of the
  Estimators for Autoregressive Time Series With a Unit Root"* (JASA) — the stationarity
  test on OLS residuals (the Engle-Granger second step), implemented via
  `statsmodels.tsa.stattools.adfuller`.

- **HC3 heteroskedasticity-robust OLS.** MacKinnon & White (1985), *"Some Heteroskedasticity
  Consistent Covariance Matrix Estimators with Improved Finite Sample Properties"* (Journal
  of Econometrics) — the robust SE used in the levels regressions.

## Data sources

- **Yahoo Finance BTC-USD daily bars** (via `yfinance`), full history available from
  2014-10-16 onward. Fetched 2026-06-12; content fingerprint `3a28b8d50d67`. The BTC
  S2F series is computed *deterministically* from the protocol's halving schedule (no
  estimation), so it is reproducible without any data download.

## Related desk studies

- **[Study 70 — Digital-Gold](../../70-digital-gold/)**: is BTC a hedge against equity
  drawdowns? — the other major BTC-as-alternative-asset claim, same empirical honesty.
- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: daily-bar 50/200 MA crossover —
  the same 'simple model, spuriously good in-sample, fails OOS' family.
- **[Study 48 — Groundhog](../../48-groundhog/)** and
  **[Study 42 — Last-Call](../../42-last-call/)**: other claims that rely on a single
  pattern in a non-stationary series — similar spuriousness hazard.
