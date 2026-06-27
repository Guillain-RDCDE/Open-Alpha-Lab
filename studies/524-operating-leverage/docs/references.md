# References & literature map — Study 524 (Operating-Leverage)

## The foundational paper

- **Novy-Marx, R. (2011).** *Operating leverage.* Review of Finance, 15(1), 103–134. The
  canonical operating-leverage paper. Novy-Marx shows that a firm's *fixed* operating costs
  generate operating leverage: a firm whose operating costs are large relative to its asset
  base behaves like a levered claim on demand, so its profits — and equity — are riskier in
  bad states. Sorting on operating costs scaled by assets, the high-operating-leverage firms
  earn higher average returns, and the measure explains a large fraction of the value
  premium (book-to-market is correlated with operating leverage). This is the signal this
  study replicates: OL = OperatingCosts / Assets, long high, short low.

## Why operating leverage carries a premium

- **Carlson, M., Fisher, A., & Giammarino, R. (2004).** *Corporate investment and asset
  price dynamics: implications for the cross-section of returns.* Journal of Finance, 59(6),
  2577–2603. A real-options / production-economy model in which operating leverage drives
  the value premium: firms with high fixed costs (often value firms with under-utilised
  assets) have riskier cash flows and command higher expected returns. The theoretical
  backbone for why operating leverage should be priced.

- **Zhang, L. (2005).** *The value premium.* Journal of Finance, 60(1), 67–103. A
  q-theoretic model linking costly reversibility and operating inflexibility (a sibling of
  operating leverage) to the value premium — value firms are burdened with unproductive
  capital and high fixed costs, making them riskier in downturns.

- **Gulen, H., Xing, Y., & Zhang, L. (2011).** *Value versus growth: time-varying expected
  stock returns.* Financial Management, 40(2), 381–407. Documents that the value premium —
  and the operating-leverage channel that underlies it — is strongly counter-cyclical,
  largest in recessions when high-fixed-cost firms are most exposed.

## Distinct from financial leverage

- **Bhandari, L. C. (1988).** *Debt/equity ratio and expected common stock returns: empirical
  evidence.* Journal of Finance, 43(2), 507–528. The classic *financial*-leverage (debt) and
  returns paper. Operating leverage (fixed-cost structure of operations) is conceptually and
  empirically distinct from financial leverage (balance-sheet debt); this desk tests the
  financial-leverage anomaly separately in [Study 154 — Leverage-Anomaly](../../154-leverage-anomaly/).

- **Mandelker, G. N., & Rhee, S. G. (1984).** *The impact of the degrees of operating and
  financial leverage on systematic risk of common stock.* Journal of Financial and
  Quantitative Analysis, 19(1), 45–57. Shows operating leverage and financial leverage are
  *separate* contributors to systematic (beta) risk — motivating a sort on operating costs
  rather than debt.

## Subsequent evidence and attenuation

- **Hou, K., Xue, C., & Zhang, L. (2020).** *Replicating anomalies.* Review of Financial
  Studies, 33(5), 2019–2133. Many accounting-based anomalies, including value-adjacent and
  operating-cost measures, attenuate substantially out of the original sample, are weaker
  among large caps, and shrink after publication — consistent with the absent effect on a
  large-cap survivor basket.

- **McLean, R. D., & Pontiff, J. (2016).** *Does academic research destroy stock return
  predictability?* Journal of Finance, 71(1), 5–32. Documents ~58% post-publication decay
  for the average anomaly; operating leverage (published 2011) is exposed to that decay.

## Survivorship bias and data limitations

- **Kothari, S. P., Sabino, J., & Zach, T. (2005).** *Implications of survival and data
  trimming for tests of market efficiency.* Journal of Accounting and Economics, 39(1),
  129–161. The basket here covers only large-cap names still trading in 2026. Because
  operating leverage is a *risk* premium that pays precisely in the bad states where
  high-fixed-cost firms fail, deleting the failures biases a real-tape test toward finding
  *no* premium — the surviving high-OL names are exactly those whose demand did not collapse.

- **Data depth.** Yahoo Finance statement endpoints expose only ~5 fiscal years; this study
  instead pulls **SEC EDGAR companyfacts** (`CostOfGoodsAndServicesSold`/`CostOfRevenue`,
  `SellingGeneralAndAdministrativeExpense`, `Assets`) to obtain ~13 years of annual
  fundamentals, yielding 17 stampable cross-sectional sort years. Cost-of-revenue tags vary
  by filer; the operating-cost measure falls back to a total-operating-cost tag
  (`CostsAndExpenses`/`OperatingExpenses`) or SG&A alone when the COGS/SG&A split is missing.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) — the
  one-sample inference on the short annual hedge series.
- **Reporting-lag discipline.** OL from fiscal year y drives a 12-month return beginning a
  conservative 4 months after the fiscal-year-end (when the 10-K is public), entered one
  trading day later — the same single-execution-lag discipline used across the desk's
  fundamental sorts (Studies 122, 153, 154, 244, 523).

## Related desk studies

- **[Study 154 — Leverage-Anomaly](../../154-leverage-anomaly/)**: the *financial*-leverage
  (balance-sheet debt) channel — the sibling Bhandari (1988) sort that 524 is deliberately
  distinct from.
- **[Study 244 — Asset-Growth](../../244-asset-growth/)**: the investment/asset-growth
  channel of the value family — None/Mirage on the survivor panel.
- **[Study 122 — Gross-Profitability](../../122-gross-profitability/)**: Novy-Marx (2013)'s
  *other* famous accounting factor (gross profits / assets), built on the same desk
  infrastructure (rolling sort, equal-weight, HAC inference).
- **[Study 523 — Investment-To-Assets](../../523-investment-to-assets/)**: the capex channel
  of the investment anomaly — same EDGAR + Yahoo plumbing, also None/Mirage.
