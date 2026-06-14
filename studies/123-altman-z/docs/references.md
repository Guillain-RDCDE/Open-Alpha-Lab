# References & literature map — Study 123 (Altman-Z)

## The model under test

- **Altman, E.I. (1968).** *Financial Ratios, Discriminant Analysis and the Prediction
  of Corporate Bankruptcy.* Journal of Finance, 23(4), 589–609. The founding paper:
  five accounting ratios combined into a discriminant score to predict bankruptcy
  within two years. Original sample of 66 manufacturing firms, 33 bankrupt and 33 not.
  Thresholds: Z < 1.81 (distress), 1.81–2.99 (grey), Z ≥ 2.99 (safe). This study
  asks whether Z also predicts **equity returns**, not just default — a distinct question.

- **Altman, E.I. (1983, updated 2000).** *Corporate Financial Distress and Bankruptcy.*
  Wiley. Extended to non-manufacturing firms (Z' score) and private firms (Z''). The
  public-company original-coefficient model (used here) is the most widely cited.

- **Altman, E.I. & Hotchkiss, E. (2006).** *Corporate Financial Distress and Bankruptcy.*
  3rd ed., Wiley. Reviews the model's forecasting power over four decades and the
  "distress puzzle" literature: low-Z firms empirically earn *lower* equity returns,
  the opposite of what a risk premium story would predict.

## The distress-risk debate

- **Dichev, I.D. (1998).** *Is the Risk of Bankruptcy a Systematic Risk?* Journal of
  Finance, 53(3), 1131–1147. Documents the distress puzzle empirically: stocks of
  high-bankruptcy-risk firms (measured by Altman Z and Ohlson O-score) **underperform**,
  contradicting the risk-premium prediction. A landmark paper this study replicates in a
  survivorship-biased setting.

- **Campbell, J.Y., Hilscher, J. & Szilagyi, J. (2008).** *In Search of Distress Risk.*
  Journal of Finance, 63(6), 2899–2939. Updates Dichev with a dynamic logit model for
  failure probability; confirms that distressed firms earn low returns. The distress
  puzzle is robust to a broader distress measure and longer sample.

- **Griffin, J.M. & Lemmon, M.L. (2002).** *Book-to-Market Equity, Distress Risk, and
  Stock Returns.* Journal of Finance, 57(5), 2317–2336. Shows that the negative return–
  distress relationship is concentrated in small stocks with low B/M ratios; partly
  absorbed by size and value factors.

## Theoretical frameworks

- **Fama, E.F. & French, K.R. (1996).** *Multifactor Explanations of Asset Pricing
  Anomalies.* Journal of Finance, 51(1), 55–84. The three-factor model interprets
  high B/M (often distressed firms) as compensation for distress risk — but this
  requires distressed firms to *earn more*, not less. The Dichev / Campbell finding
  contradicts this interpretation directly.

- **Vassalou, M. & Xing, Y. (2004).** *Default Risk in Equity Returns.* Journal of
  Finance, 59(2), 831–868. Uses Merton (1974) KMV-style expected default frequency
  (EDF) and finds a positive risk premium — in contrast to Dichev. The discrepancy
  may reflect small-firm / micro-cap dynamics absent in the S&P 500 universe here.

- **Garlappi, L. & Yan, H. (2011).** *Financial Distress and the Cross-Section of
  Equity Returns.* Journal of Finance, 66(3), 789–822. Provides a structural model
  explaining why distressed firms can earn low returns: shareholder recovery value
  and renegotiation power shift the risk to creditors, reducing the equity risk premium.

## Related desk studies

- **[Study 52 — Smoke-Screen](../../52-smoke-screen/)**: accruals anomaly from the
  same EDGAR cache — the same survivorship-biased data infrastructure.
- **[Study 65 — Scorecard](../../65-scorecard/)**: Piotroski F-score, the other
  classic fundamental-health composite signal from these data.
- **[Study 44 — Growth-Spurt](../../44-growth-spurt/)**: revenue growth as a stand-alone
  EDGAR factor — another single-concept pull from the same panel.
- **[Study 51 — Blue-Chip](../../51-blue-chip/)**: balance-sheet quality as a factor,
  related to the liquidity/leverage components of the Z-score.

## Method lineage

- **Newey-West HAC t-stat.** Newey, W.K. & West, K.D. (1987). *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.*
  Econometrica, 55(3), 703–708. Used in ``strategy.summarize`` and
  ``quantlab.analytics.mean_tstat_hac`` for inference on the 16-year annual hedge.
- **Block-bootstrap Sharpe CI.** Politis, D.N. & Romano, J.P. (1994). *The Stationary
  Bootstrap.* JASA, 89(428), 1303–1313. Used in ``quantlab.stats.sharpe_ci_bootstrap``
  for the annualised Sharpe confidence interval.

## Data sources

- **EDGAR shared caches** (``_cache/_edgar_*.parquet``): Assets, AssetsCurrent,
  LiabilitiesCurrent, RetainedEarningsAccumulatedDeficit, OperatingIncomeLoss,
  Liabilities, Revenues, WeightedAverageNumberOfDilutedSharesOutstanding — annual
  10-K FY values, 2006–2026, current S&P 500 tickers (survivorship-biased).
- **yfinance monthly prices**: December month-end adjusted close for market-cap
  computation; annual returns via ``_edgar_yrret.parquet``.
