# References & literature map — Study 230 (Ohlson O-score)

## The model under test

- **Ohlson, J.A. (1980).** *Financial Ratios and the Probabilistic Prediction
  of Bankruptcy.* Journal of Accounting Research, 18(1), 109–131. The founding
  paper: a logit model combining nine accounting ratios into a composite distress
  probability score. Estimated on 105 bankrupt and 2,058 non-bankrupt US firms
  (1970–1976). Unlike Altman's (1968) discriminant approach, the O-score produces
  a probability interpretation. This study tests whether O-score rank-sorts
  equity returns, a distinct question from bankruptcy prediction.

- **Altman, E.I. (1968).** *Financial Ratios, Discriminant Analysis and the
  Prediction of Corporate Bankruptcy.* Journal of Finance, 23(4), 589–609.
  The companion model — five-variable linear discriminant. Tested in desk
  study 123 (Altman-Z). The O-score is conceptually motivated by the same
  distress literature; comparing the two is the key motivation for this study.

## The distress-risk debate

- **Dichev, I.D. (1998).** *Is the Risk of Bankruptcy a Systematic Risk?*
  Journal of Finance, 53(3), 1131–1147. Documents the distress puzzle using
  both the Altman Z-score and the Ohlson O-score: stocks with high bankruptcy
  risk **underperform**, contradicting the risk-premium prediction. This study
  attempts to replicate Dichev's O-score finding on a survivorship-biased
  S&P 500 panel.

- **Campbell, J.Y., Hilscher, J. & Szilagyi, J. (2008).** *In Search of
  Distress Risk.* Journal of Finance, 63(6), 2899–2939. Updates the distress
  puzzle with a dynamic logit model using market and accounting variables.
  Confirms that distressed firms earn low returns; robust to the Altman and
  Ohlson measures. The Campbell-Hilscher-Szilagyi (CHS) model is a direct
  descendant of the O-score tradition.

- **Griffin, J.M. & Lemmon, M.L. (2002).** *Book-to-Market Equity, Distress
  Risk, and Stock Returns.* Journal of Finance, 57(5), 2317–2336. Finds that
  the negative distress-return relationship is concentrated among firms with
  low book-to-market ratios, suggesting the puzzle is linked to growth stocks
  being most financially distressed.

- **Grice, J.S. & Dugan, M.T. (2003).** *Re-Estimations of the Zmijewski and
  Ohlson Bankruptcy Prediction Models.* Advances in Accounting, 20, 77–93.
  Re-estimates the O-score on modern samples; finds performance degradation
  outside the original period, consistent with the broader finding that
  distress models' equity-return predictive power decays out of sample.

## Theoretical frameworks

- **Fama, E.F. & French, K.R. (1996).** *Multifactor Explanations of Asset
  Pricing Anomalies.* Journal of Finance, 51(1), 55–84. The three-factor model
  interprets high B/M (often distressed) firms as earning risk compensation —
  but requires distressed firms to earn *more*, not less. The Dichev/Ohlson
  puzzle challenges this.

- **Garlappi, L. & Yan, H. (2011).** *Financial Distress and the Cross-Section
  of Equity Returns.* Journal of Finance, 66(3), 789–822. Provides a structural
  model: shareholder renegotiation power in distress reduces the equity risk
  premium, explaining why high-O firms may not earn extra return.

- **Vassalou, M. & Xing, Y. (2004).** *Default Risk in Equity Returns.* Journal
  of Finance, 59(2), 831–868. Uses Merton KMV expected default frequency and
  finds a positive risk premium — inconsistent with Dichev. The discrepancy
  may reflect the broader, unbiased universe (including small caps) vs the S&P
  500 panel here.

## Related desk studies

- **[Study 123 — Altman-Z](../../123-altman-z/)**: the five-variable Altman
  Z-score tested on the same EDGAR universe. The direct comparison that motivates
  this study — the hook question is explicitly "does O-score price distress any
  better than Altman-Z?"
- **[Study 52 — Smoke-Screen](../../52-smoke-screen/)**: accruals anomaly from
  the same EDGAR infrastructure.
- **[Study 65 — Scorecard](../../65-scorecard/)**: Piotroski F-score, the
  other classic fundamental-health composite from these data.
- **[Study 139 — Magic Formula](../../139-magic-formula/)**: quality-and-value
  composite; related concept to distress screening.

## Method lineage

- **Newey-West HAC t-stat.** Newey, W.K. & West, K.D. (1987). *A Simple,
  Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent
  Covariance Matrix.* Econometrica, 55(3), 703–708. Used in ``strategy.summarize``
  and ``quantlab.analytics.mean_tstat_hac`` for inference on the 16-year annual
  hedge.
- **Block-bootstrap Sharpe CI.** Politis, D.N. & Romano, J.P. (1994). *The
  Stationary Bootstrap.* JASA, 89(428), 1303–1313. Used in
  ``quantlab.stats.sharpe_ci_bootstrap`` for the annualised Sharpe CI.

## Data sources

- **EDGAR shared caches** (``_cache/_edgar_*.parquet``): Assets, AssetsCurrent,
  LiabilitiesCurrent, Liabilities, NetIncomeLoss,
  NetCashProvidedByUsedInOperatingActivities — annual 10-K FY values, 2006–2026,
  current S&P 500 tickers (survivorship-biased).
- **_edgar_yrret.parquet**: annual price returns for the same S&P 500 universe.
