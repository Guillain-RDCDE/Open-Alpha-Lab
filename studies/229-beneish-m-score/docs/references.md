# References & literature map — Study 229 (Beneish M-score)

## The model under test

- **Beneish, M.D. (1999).** *The Detection of Earnings Manipulation.* Financial Analysts
  Journal, 55(5), 24–36. The founding paper: eight accounting-ratio components (DSRI, GMI,
  AQI, SGI, DEPI, SGAI, TATA, LVGI) combined into a probit discriminant score. The
  calibration sample was 74 manipulators vs. 2,332 non-manipulators (GAAP-violation firms
  from SEC enforcement actions, 1982–1988). Threshold M > −1.78 flags likely manipulators.
  This study tests whether M also predicts **equity returns** — a distinct question from
  *ex-post* manipulation detection.

- **Beneish, M.D., Lee, C.M.C. & Nichols, D.C. (2013).** *Earnings Manipulation and
  Expected Stock Returns.* Working Paper, Indiana University. Shows that a long/short
  portfolio based on M-score earned ~14%/yr in the 1993–2003 period before publication.
  Post-publication decay is consistent with the pattern for discovered anomalies; our
  study tests the 2009–2023 window.

## Theoretical frameworks

- **Ball, R. & Brown, P. (1968).** *An Empirical Evaluation of Accounting Income Numbers.*
  Journal of Accounting Research, 6(2), 159–178. Foundational work on the information
  content of earnings — the premise that accounting distortions (targeted by Beneish) carry
  economic information that markets price only partially.

- **Sloan, R.G. (1996).** *Do Stock Prices Fully Reflect Information in Accruals and Cash
  Flows About Future Earnings?* The Accounting Review, 71(3), 289–315. The accruals
  anomaly: high-accrual firms (which inflate reported earnings) earn lower subsequent
  returns. TATA (accruals / total assets) in the M-score captures this channel directly;
  this study is the closest ancestor of the TATA component.

- **Dechow, P.M., Sloan, R.G. & Sweeney, A.P. (1995).** *Detecting Earnings Management.*
  The Accounting Review, 70(2), 193–225. Evaluates models of earnings management detection;
  the Jones model and its variants underpin the AQI and TATA construction.

## The manipulation-return link

- **Beneish, M.D. & Vargus, M.E. (2002).** *Insider Trading, Earnings Quality, and Accrual
  Mispricing.* The Accounting Review, 77(4), 755–791. Insiders sell heavily in high-accrual,
  high-M-score firms — consistent with the score detecting overvaluation, not just
  manipulation risk.

- **Cecchini, M., Aytug, H., Koehler, G.J. & Pathak, P. (2010).** *Making Words Work: Using
  Financial Text as a Predictor of Financial Events.* Decision Support Systems, 50(1),
  164–175. Extends the Beneish approach with textual features from MD&A disclosures.

- **Alali, F. & Romero, S. (2013).** *Characteristics of Failed US Commercial Banks: An
  Exploratory Study.* Accounting & Finance, 53(4), 1149–1174. Shows M-score elevated
  before bank failures; financial-sector firms are excluded from our panel (many report
  no GrossProfit and use neutral GMI/SGAI fallbacks).

## Survivorship and selection effects

- **Dichev, I.D. (1998).** *Is the Risk of Bankruptcy a Systematic Risk?* Journal of
  Finance, 53(3), 1131–1147. The "distress puzzle" closest to our finding: high-distress
  firms do not earn higher returns in a survivorship-biased large-cap panel. The
  manipulation puzzle is the same conceptual failure: high-M firms in the S&P 500
  survivor set are not actually the manipulators Beneish calibrated on.

- **McLean, R.D. & Pontiff, J. (2016).** *Does Academic Publication of Anomalies Destroy
  Mispricing?* Review of Financial Studies, 29(1), 267–297. Anomaly returns decay after
  publication on average by ~58%. Beneish (1999) is now 25+ years old; the 2013 Beneish–
  Lee–Nichols study brought it to practitioner attention. Post-publication decay is a
  likely contributor to the absent signal we find.

## Related desk studies

- **[Study 123 — Altman-Z](../../123-altman-z/)**: the structural sibling — a composite
  fundamental score tested against the same EDGAR panel. Finds the same None/Mirage
  result on the S&P 500 survivor universe for a similar reason: the interesting failures
  are absent from the data.
- **[Study 52 — Smoke-Screen](../../52-smoke-screen/)**: accruals anomaly from EDGAR
  — the TATA component of M-score in isolation.
- **[Study 65 — Scorecard](../../65-scorecard/)**: Piotroski F-score, another 9-variable
  composite fundamental signal from EDGAR data.

## Method lineage

- **Newey-West HAC t-stat.** Newey, W.K. & West, K.D. (1987). *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.*
  Econometrica, 55(3), 703–708. Used in ``strategy.summarize`` for inference on the
  15-year annual hedge.
- **Block-bootstrap Sharpe CI.** Politis, D.N. & Romano, J.P. (1994). *The Stationary
  Bootstrap.* JASA, 89(428), 1303–1313. Used in ``quantlab.stats.sharpe_ci_bootstrap``
  for the annualised Sharpe confidence interval.

## Data sources

- **EDGAR shared caches** (``_cache/_edgar_*.parquet``): Assets, AssetsCurrent,
  LiabilitiesCurrent, LongTermDebtNoncurrent, Revenues, GrossProfit, OperatingIncomeLoss,
  NetIncomeLoss, NetCashProvidedByUsedInOperatingActivities — annual 10-K FY values.
- **EDGAR study-local caches** (``studies/229-beneish-m-score/_cache/``):
  AccountsReceivableNetCurrent and PropertyPlantAndEquipmentNet — fetched from
  data.sec.gov on first run, 161 common tickers.
- **_edgar_yrret.parquet**: annual returns from yfinance, desk-shared cache.
