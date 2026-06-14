# References & literature map — Study 122 (Gross-Profitability)

## The primary claim under test

- **Novy-Marx, R. (2013).** "The Other Side of Value: The Gross Profitability Premium."
  *Journal of Financial Economics*, 108(1), 1–28. The founding paper: GrossProfit / Assets
  (GP/A) predicts the cross-section of stock returns as reliably as book-to-market. High-GP/A
  firms are "quality" firms and earn a premium over low-GP/A firms; the author argues
  profitability and value are both driven by the same rational pricing of growth options.
  The strategy: annually sort on GP/A, long top quintile, short bottom quintile. Reported
  factor Sharpe ~0.5–0.7 on a broad US universe (1963–2010).

## Why the effect should exist — the theoretical backbone

- **Fama, E. F. & French, K. R. (2006).** "Profitability, Investment, and Average Returns."
  *Journal of Financial Economics*, 82(3), 491–518. Establishes the theoretical link:
  holding book-to-market fixed, more profitable firms should earn higher expected returns
  (discounted cash flow identity). GP/A operationalises this.
- **Fama, E. F. & French, K. R. (2015).** "A Five-Factor Asset Pricing Model." *Journal of
  Financial Economics*, 116(1), 1–22. Adds a profitability factor (RMW — robust minus weak)
  to the Fama-French model; GP/A is the empirical anchor for that factor. Positive exposure
  to RMW is systematically rewarded in the cross-section.

## Subsequent evidence, replication, and attenuation

- **Hou, K., Xue, C., & Zhang, L. (2020).** "Replicating Anomalies." *Review of Financial
  Studies*, 33(5), 2019–2133. Replicates ~400 anomalies; gross profitability survives
  replication but with attenuated magnitudes, especially post-publication.
- **McLean, R. D. & Pontiff, J. (2016).** "Does Academic Research Destroy Stock Return
  Predictability?" *Journal of Finance*, 71(1), 5–32. Documents that cross-sectional
  return predictors weaken by ~32% after academic publication; gross profitability, first
  published in 2013, is not immune.
- **Chordia, T., Subrahmanyam, A., & Tong, Q. (2014).** "Have Capital Market Anomalies
  Attenuated in the Recent Era of High Liquidity and Investor Attention?" *Journal of
  Accounting and Economics*, 58(1), 41–58. Shows that improved arbitrage capacity since
  2000 compresses return anomalies.

## Survivorship bias and universe construction

- **Shumway, T. & Warther, V. A. (1999).** "The Delisting Bias in CRSP's Nasdaq Data and
  Its Implications for the Size Effect." *Journal of Finance*, 54(6), 2361–2379. Delistings
  are not random — failed firms (a natural short candidate for quality factors) have
  disproportionately large negative returns. Excluding them biases estimates upward.
- **Linnainmaa, J. T. & Roberts, M. R. (2018).** "The History of the Cross-Section of
  Stock Returns." *Review of Financial Studies*, 31(7), 2606–2649. Shows that many
  accounting-based anomalies, including profitability sorts, are weaker or absent in the
  pre-COMPUSTAT era; survivorship and selection effects matter.

## Trading costs and implementability

- **Novy-Marx, R. & Velikov, M. (2016).** "A Taxonomy of Anomalies and Their Trading
  Costs." *Review of Financial Studies*, 29(1), 104–147. Annual quality-factor strategies
  have modest turnover (~30–50%/yr at large-cap), making them among the *more* implementable
  factor strategies; but the net alpha on a large-cap universe is thin.
- **Frazzini, A., Israel, R., & Moskowitz, T. J. (2015).** "Trading Costs of Asset Pricing
  Anomalies." Working paper. Estimates that quality (profitability) strategies can survive
  realistic institutional trading costs on large-cap universes, but the margin is narrow.

## Method lineage (the desk's shared engine)

- **Newey, W. K. & West, K. D. (1987).** "A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica*,
  55(3), 703–708. The HAC long-run variance estimator in
  [`strategy.summary`](../gross_profitability/strategy.py).
- EDGAR XBRL fundamentals: SEC data accessed via the desk's shared
  `_edgar_GrossProfit.parquet` and `_edgar_Assets.parquet` caches; annual returns from
  `_edgar_yrret.parquet` (Yahoo Finance monthly prices, calendar-year compounded).

## Related desk studies

- **[Study 52 — Smoke-Screen](../../52-smoke-screen/)**: Sloan's accruals anomaly — another
  EDGAR-based annual fundamental sort using the same infrastructure (NetIncomeLoss, CFO, Assets).
- **[Study 65 — Scorecard](../../65-scorecard/)**: Piotroski's F-score — a composite quality
  signal using nine fundamental signals; directly comparable methodology and universe.
- **[Study 44 — Growth-Spurt](../../44-growth-spurt/)**: revenue growth factor — complements
  profitability in the Fama-French five-factor world.
- **[Study 56 — Tide-Table](../../56-tide-table/)**: CAPE-based valuation timing — the
  "other side" of value that Novy-Marx contrasts against profitability.
