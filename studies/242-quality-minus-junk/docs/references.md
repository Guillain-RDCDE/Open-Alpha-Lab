# References & literature map — Study 242 (Quality-Minus-Junk)

## The primary claim under test

- **Asness, C., Frazzini, A. & Pedersen, L. H. (2019).** "Quality Minus Junk." *Review of
  Accounting Studies*, 24(1), 34–112. The defining paper: a composite quality score
  (profitability + growth + safety + payout) long-shorts across the US market, MSCI World,
  and across asset classes. Reported Sharpe ~1.3 on the US long-short (1957–2016). Four
  pillars: Profitability (ROE, ROA, GP/A, net margin, CFO/A, accruals); Growth (5-year
  change in profitability z-score); Safety (BAB beta, leverage, volatility, Z-score credit);
  Payout (issuance, buyback, dividend). We approximate with three EDGAR-available pillars.

## Why the effect should exist — the theoretical backbone

- **Fama, E. F. & French, K. R. (2015).** "A Five-Factor Asset Pricing Model." *Journal of
  Financial Economics*, 116(1), 1–22. The RMW (robust-minus-weak profitability) and CMA
  (conservative-minus-aggressive investment) factors partially overlap with QMJ's profitability
  and safety pillars. Both are theoretically motivated by discounted cash flow accounting.
- **Gordon, M. J. (1962).** *The Investment, Financing, and Valuation of the Corporation.*
  Irwin. The DDM identity that profitable, low-growth firms should earn higher required
  returns for their level of risk — the DGM backbone of quality pricing.
- **Novy-Marx, R. (2013).** "The Other Side of Value: The Gross Profitability Premium."
  *Journal of Financial Economics*, 108(1), 1–28. Establishes that GP/A alone predicts the
  cross-section; QMJ extends this to a multi-pillar composite. See also Study 122.

## Subsequent evidence, replication, and attenuation

- **Hou, K., Xue, C. & Zhang, L. (2020).** "Replicating Anomalies." *Review of Financial
  Studies*, 33(5), 2019–2133. Replicates ~400 anomalies including quality proxies; profitability
  and investment factors survive, but with attenuated post-publication magnitudes.
- **McLean, R. D. & Pontiff, J. (2016).** "Does Academic Research Destroy Stock Return
  Predictability?" *Journal of Finance*, 71(1), 5–32. Documents ~32% post-publication decay
  for cross-sectional predictors; AQR's QMJ has been widely marketed since 2013, suggesting
  the remaining premium is the hardest to arbitrage away.
- **Israel, R., Laursen, K. & Richardson, S. (2021).** "Is (Systematic) Value Investing
  Dead?" *Journal of Portfolio Management*, 47(2), 1–23. Discusses crowding of systematic
  quality factors in the 2018-2020 period and subsequent reversal.
- **Blitz, D. & Hanauer, M. X. (2021).** "Resurrecting the Value Premium." *Journal of
  Portfolio Management*, 47(2), 63–81. Shows value and quality factors have become more
  correlated as both attract systematic capital, compressing independent premia.

## Survivorship bias and universe construction

- **Shumway, T. (1997).** "The Delisting Bias in CRSP Data." *Journal of Finance*, 52(1),
  327–340. Failed companies (natural short candidates for quality factors) have large
  negative returns on delisting; excluding them biases hedge estimates upward.
- **Linnainmaa, J. T. & Roberts, M. R. (2018).** "The History of the Cross-Section of
  Stock Returns." *Review of Financial Studies*, 31(7), 2606–2649. Many accounting-based
  anomalies are weaker in non-biased samples and in out-of-sample periods.

## Trading costs and implementability

- **Novy-Marx, R. & Velikov, M. (2016).** "A Taxonomy of Anomalies and Their Trading
  Costs." *Review of Financial Studies*, 29(1), 104–147. Quality/profitability strategies
  have low turnover (annual rebalance) and survive realistic costs on large-cap universes,
  but the net alpha margin is thin.
- **Frazzini, A., Israel, R. & Moskowitz, T. J. (2015).** "Trading Costs of Asset Pricing
  Anomalies." Working paper. Estimates that quality strategies can survive institutional
  trading costs, but with a narrow margin on large-cap universes.

## Method lineage (the desk's shared engine)

- **Newey, W. K. & West, K. D. (1987).** "A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica*,
  55(3), 703–708. The HAC long-run variance estimator in
  [`strategy.summary`](../quality_minus_junk/strategy.py).
- EDGAR XBRL fundamentals: SEC data accessed via the desk's shared caches
  (`_edgar_GrossProfit.parquet`, `_edgar_Assets.parquet`, `_edgar_NetIncomeLoss.parquet`,
  `_edgar_Revenues.parquet`, `_edgar_StockholdersEquity.parquet`,
  `_edgar_NetCashProvidedByUsedInOperatingActivities.parquet`); annual returns from
  `_edgar_yrret.parquet` (Yahoo Finance monthly prices, calendar-year compounded).

## Related desk studies

- **[Study 122 — Gross-Profitability](../../122-gross-profitability/)**: Novy-Marx GP/A
  factor — the profitability pillar of QMJ in isolation; same EDGAR infrastructure, directly
  comparable setup (Weak/Fragile verdict).
- **[Study 52 — Smoke-Screen](../../52-smoke-screen/)**: Sloan's accruals anomaly — another
  EDGAR-based accounting quality signal, same universe.
- **[Study 65 — Scorecard](../../65-scorecard/)**: Piotroski's F-score — composite quality
  signal with nine binary fundamental signals; direct competitor to QMJ.
- **[Study 120 — Excess-CAPE-Yield](../../120-excess-cape-yield/)**: the valuation
  counterpart — value meets quality in the Shiller framework.
- **[Study 138 — Magic Formula](../../138-magic-formula/)**: Greenblatt's rank-sum of
  earnings yield × return on capital — a simpler quality-value composite.
