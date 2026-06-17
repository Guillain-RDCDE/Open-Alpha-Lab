# References & literature map -- Study 232 (Mohanram G-score)

## The primary claim under test

- **Mohanram, P. S. (2005).** "Separating Winners from Losers among Low Book-to-Market
  Stocks using Financial Statement Analysis." *Review of Accounting Studies*, 10(2-3),
  133--170. The founding paper: the G-score packages 8 binary accounting signals into a
  composite score for growth/glamour (low book-to-market) stocks. High-G firms are
  predicted to earn higher future returns than low-G firms. The G-score is the growth
  counterpart to Piotroski's F-score for value stocks. Reported long-short returns of
  ~10%/yr on a broad US universe (1979--2001).

## G-score signal structure -- the 8 components

The original Mohanram (2005) G-score:
  - **G1 ROA:** NetIncome / Assets > industry median
  - **G2 CFO:** Cash flow from operations / Assets > industry median
  - **G3 Accruals:** (NI - CFO) / Assets < industry median (lower = higher cash quality)
  - **G4 ROA variability:** rolling std of ROA < industry median (earnings stability)
  - **G5 Sales variability:** rolling std of Sales/Assets < industry median (growth stability)
  - **G6 R&D intensity:** R&D expense / Assets > industry median (innovation investment)
  - **G7 Capital expenditure:** CapEx growth > 0 (capacity investment)
  - **G8 Advertising intensity:** Advertising / Assets > industry median (brand investment)

This desk implementation substitutes G6 -> revenue growth > 0, and G7 -> asset-turnover
growth > 0, because R&D and advertising concepts are not in the shared EDGAR cache.

## Why the effect should exist -- the theoretical backbone

- **Mohanram, P. S. (2005).** The original thesis: growth stocks are systematically
  mispriced around optimistic analyst forecasts. Fundamental strength (profitability,
  stability, conservative investment) identifies growth stocks with real competitive
  advantage vs glamour stocks priced on hype alone.
- **Piotroski, J. D. (2000).** "Value Investing: The Use of Historical Financial Statement
  Information to Separate Winners from Losers." *Journal of Accounting Research*, 38,
  1--41. The F-score counterpart for value stocks; the G-score is the direct extension
  to growth stocks. See Study 65 (Scorecard) for the desk's F-score teardown.
- **Sloan, R. G. (1996).** "Do Stock Prices Fully Reflect Information in Accruals and
  Cash Flows about Future Earnings?" *Accounting Review*, 71(3), 289--315. The accruals
  anomaly underlies G3; cash earnings are higher quality than accrual earnings.

## Subsequent evidence, replication, and attenuation

- **Hou, K., Xue, C., & Zhang, L. (2020).** "Replicating Anomalies." *Review of Financial
  Studies*, 33(5), 2019--2133. Documents that many accounting-based composites, including
  growth-stock screens, weaken or fail to replicate out of sample. Universe construction
  and data availability matter greatly.
- **McLean, R. D. & Pontiff, J. (2016).** "Does Academic Research Destroy Stock Return
  Predictability?" *Journal of Finance*, 71(1), 5--32. Post-publication attenuation
  averages ~32% for accounting anomalies; the G-score was published in 2005 and has
  been incorporated into systematic strategies since.
- **Fama, E. F. & French, K. R. (2008).** "Dissecting Anomalies." *Journal of Finance*,
  63(4), 1653--1678. Shows that many composite signals work in small-caps but attenuate
  substantially in large-caps where arbitrage is cheapest.

## Survivorship bias and universe construction

- **Shumway, T. (1997).** "The Delisting Return Puzzle: A Fix and Its Implications."
  *Journal of Finance*, 52(6), 2539--2547. Firms that get delisted (often the worst
  growth-stock busts) have large negative returns that are missing from the S&P 500
  survivor panel. Excluding them biases long-leg estimates upward.
- **Linnainmaa, J. T. & Roberts, M. R. (2018).** "The History of the Cross-Section of
  Stock Returns." *Review of Financial Studies*, 31(7), 2606--2649. Many accounting-based
  anomalies are artefacts of the COMPUSTAT survivorship and coverage period.

## Trading costs and implementability

- **Novy-Marx, R. & Velikov, M. (2016).** "A Taxonomy of Anomalies and Their Trading
  Costs." *Review of Financial Studies*, 29(1), 104--147. Annual fundamental sorts have
  modest turnover and can survive realistic costs -- but the G-score on large-cap survivors
  has a negative expected return before costs.

## Method lineage (the desk's shared engine)

- **Newey, W. K. & West, K. D. (1987).** "A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica*,
  55(3), 703--708. The HAC long-run variance estimator in
  [`strategy.summary`](../mohanram_g_score/strategy.py).
- EDGAR XBRL fundamentals: SEC data accessed via the desk's shared EDGAR caches;
  annual returns from `_edgar_yrret.parquet` (Yahoo Finance monthly, calendar-year
  compounded).

## Related desk studies

- **[Study 65 -- Scorecard](../../65-scorecard/)**: Piotroski's F-score -- the value-stock
  counterpart to the G-score; same EDGAR infrastructure and quintile-sort engine.
- **[Study 122 -- Gross-Profitability](../../122-gross-profitability/)**: Novy-Marx (2013)
  GP/A factor; same EDGAR assets and NetIncome concepts, overlapping universe.
- **[Study 138 -- Magic-Formula](../../138-magic-formula/)**: Greenblatt's combination of
  earnings yield and return on capital -- another growth-quality composite.
- **[Study 52 -- Smoke-Screen](../../52-smoke-screen/)**: Sloan's accruals anomaly -- the
  theoretical basis for G3 (cash earnings quality) in the G-score.
