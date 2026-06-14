# References & literature map — Study 121 (Magic-Formula)

## The claim under test

- **Greenblatt (2005).** *The Little Book That Beats the Market* (John Wiley & Sons). The
  canonical source: rank every stock by (1) Return on Capital = EBIT / (Net Working Capital +
  Net Fixed Assets) and (2) Earnings Yield = EBIT / Enterprise Value; sum the two ranks; buy
  the top 20–30 annually and hold for a year. Greenblatt reports ~30%/yr compounded over 1988–
  2004 on a universe of US stocks with market caps above $50M. The formula is elegant because
  it jointly targets *quality* (ROC screens for durable moats) and *value* (EY screens for
  price paid per unit of earnings) — the two factors with the most robust academic support.
- **Greenblatt (2010).** *The Little Book That Still Beats the Market* (John Wiley & Sons).
  Updated version including the 2000–2009 decade; performance in the book remains strong,
  though Greenblatt notes that the formula requires patient investors willing to underperform
  for stretches of 1–3 years.

## The real effects behind the recipe — academic anchors

- **Value / Earnings Yield.** Basu (1977), *Investment Performance of Common Stocks in
  Relation to Their Price-Earnings Ratios* (Journal of Finance) — the original US evidence
  that low-P/E stocks outperform. Fama & French (1992), *The Cross-Section of Expected Stock
  Returns* (Journal of Finance) — book-to-market and size explain most of the cross-section.
  Asness, Frazzini & Pedersen (2019), *Quality Minus Junk* (Review of Accounting Studies) —
  the modern quality factor, closely related to Greenblatt's ROC axis.
- **Quality / Profitability.** Novy-Marx (2013), *The Other Side of Value: The Gross
  Profitability Premium* (Journal of Financial Economics) — gross profitability (revenue −
  COGS / assets) predicts returns with the opposite sign to book-to-market, suggesting quality
  is separately priced. Fama & French (2015), *A Five-Factor Asset Pricing Model* (Journal of
  Financial Economics) — profitability (RMW) and investment (CMA) factors extend the FF3.
- **Combined quality + value.** Piotroski (2000), *Value Investing: The Use of Historical
  Financial Statement Information to Separate Winners from Losers* (Journal of Accounting
  Research) — a closely related 9-point fundamental health score; see also
  [Study 65 — Scorecard](../../65-scorecard/). Gray & Vogel (2012), *Analyzing Valuation
  Measures: A Performance Horse Race over the Past 40 Years* (Journal of Portfolio Management)
  — EBIT-to-EV (the MF earnings yield) is one of the most predictive single value metrics.

## Why the effect weakens on large, liquid, covered stocks

- **Factor crowding and the S&P 500 universe.** Chordia, Subrahmanyam & Tong (2014),
  *Have Capital Market Anomalies Attenuated in the Recent Era of High Liquidity and Trading
  Activity?* (Journal of Accounting and Economics) — most well-documented anomalies weaken
  significantly post-publication and in highly liquid, institutional-analyst-dense names.
- **Piotroski on large caps (the inversion).** Piotroski (2000) himself notes the F-score is
  designed for book-to-market value stocks, not index blue-chips. On the S&P 500, the same
  quality signals that identify neglected winners in small caps identify *expensive* high-ROIC
  growth companies in large caps — already priced for perfection — while the "weak" names are
  often beaten-down cyclicals that rebound. Study 65 of this desk finds the F-score sign
  inverts on the S&P 500 (+22%/yr low-F vs +18%/yr high-F); the Magic Formula bottom decile
  outperforming the top is consistent with that finding.
- **Survivorship bias in academic replication.** Shumway (1997), *The Delisting Bias in
  CRSP's Nasdaq Data and Its Implications for the Size Effect* (Journal of Finance) — ignoring
  delistings and failures inflates factor returns. The EDGAR caches used here cover only
  *current* S&P 500 members projected backwards: all Enrons, Lehman Brothers, and GM
  bankruptcies are by definition absent from the panel. Positive results are upper bounds.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) — used
  for the HAC t-stat on the annual hedge returns in
  [`strategy.summary`](../magic_formula/strategy.py).
- **Reporting lag.** Chan, Jegadeesh & Lakonishok (1996), *Momentum Strategies* (Journal of
  Finance) — the convention of lagging accounting data by one full year (fundamentals from
  year y → returns year y+1) is conservative but honest: 10-Ks are typically filed within 60
  days of fiscal year-end, but the desk applies a full-year lag to avoid any look-ahead.
- **EDGAR XBRL data.** SEC EDGAR company concept API; the desk's shared
  ``_cache/_edgar_<Concept>.parquet`` frames (columns = tickers, index = fiscal year,
  values = annual 10-K FY USD values). The XBRL reporting era begins ~2007 for early large-cap
  filers and broadens to near-full S&P 500 coverage by 2010 — hence the power asymmetry
  between early and late years in the panel.

## Related desk studies

- **[Study 65 — Scorecard](../../65-scorecard/)**: Piotroski's F-score on the same S&P 500
  EDGAR panel — finds the same inversion (low-F firms beat high-F on large caps) and the same
  survivorship caveat.
- **[Study 52 — Smoke-Screen](../../52-smoke-screen/)**: accruals anomaly on EDGAR — a
  related quality/earnings-quality factor that does replicate weakly.
- **[Study 51 — Blue-Chip](../../51-blue-chip/)**: quality screens on blue-chip names — same
  universe, different metrics.
- **[Study 44 — Growth-Spurt](../../44-growth-spurt/)**: earnings-growth momentum on EDGAR —
  the profitability-change axis of quality factors.
