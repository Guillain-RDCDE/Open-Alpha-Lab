# References & literature map — Study 529 (Inventory-Growth)

## The foundational papers

- **Belo, F., & Lin, X. (2012).** *The inventory growth spread.* Review of Financial
  Studies, 25(1), 278–313. The anchor paper. Builds a production-based asset-pricing model
  in which firms with high inventory growth have *low* expected returns, and documents an
  inventory-growth spread: a low-minus-high inventory-growth portfolio earns a positive,
  significant premium in the US cross-section. Inventory growth is a real-investment signal
  closely tied to the broader investment / asset-growth factor.

- **Thomas, J. K., & Zhang, H. (2002).** *Inventory changes and future returns.* Review
  of Accounting Studies, 7(2–3), 163–187. The accounting-side anchor. Shows that abnormal
  inventory changes (inventory growth net of sales growth, scaled by assets) negatively
  predict future stock returns — firms that build inventory faster than they sell
  subsequently underperform. Motivates scaling the inventory *change* by lagged total
  assets rather than by lagged inventory.

## Related real-investment / asset-growth anomalies

- **Cooper, M. J., Gulen, H., & Schill, M. J. (2008).** *Asset growth and the cross
  section of stock returns.* Journal of Finance, 63(4), 1609–1651. The total-asset-growth
  anomaly of which inventory growth is one component. High total-asset growth → low
  returns across NYSE/AMEX/NASDAQ. Tested on this desk in
  [Study 244 — Asset-Growth](../244-asset-growth/).

- **Titman, S., Wei, K. C. J., & Xie, F. (2004).** *Capital investments and stock
  returns.* Journal of Financial and Quantitative Analysis, 39(4), 677–700. High capital
  expenditure (another real-investment expansion signal) predicts underperformance — the
  over-investment cousin of inventory build.

- **Fama, E. F., & French, K. R. (2015).** *A five-factor asset pricing model.* Journal
  of Financial Economics, 116(1), 1–22. The CMA (Conservative Minus Aggressive investment)
  factor operationalises the real-investment effect at the factor level; inventory growth
  is one of its accounting drivers.

## Why the effect might exist (the mechanism debate)

- **q-theory / rational pricing.** Firms invest (build inventory) when discount rates are
  low; low discount rates mechanically imply low expected returns. The inventory-growth
  spread is then a rational risk-premium artefact, not mispricing (Belo & Lin 2012).

- **Over-extrapolation / over-investment.** Managers and markets extrapolate demand;
  inventory builds ahead of sales that fail to materialise, and the stock disappoints as
  the build is written down or discounted (Thomas & Zhang 2002; Jensen 1986 agency costs).

- **Limits to arbitrage.** As with most accounting anomalies, the effect is strongest in
  small, illiquid, low-coverage names; on heavily-covered large-cap survivors it is
  largely arbitraged away — consistent with the flat result on this study's basket.

## Survivorship bias and the data limitation

- **Kothari, S. P., Sabino, J., & Zach, T. (2005).** *Implications of survival and data
  trimming for tests of market efficiency.* Journal of Accounting and Economics, 39(1),
  129–161. Why survivor panels bias anomaly tests: failed firms (here, over-builders that
  drowned in unsold inventory) are exactly the names that would drive the predicted sign,
  and they are absent from a current-membership basket.

- **Data depth caveat (this study).** Yahoo Finance's `balance_sheet` endpoint serves only
  ~4–5 fiscal years of annual statements per ticker. After the one-year reporting lag and
  a complete-calendar-year forward-return filter, only two usable hedge years survive. The
  original papers use decades of Compustat point-in-time data across thousands of names;
  this study cannot replicate that breadth and reports the limitation openly rather than
  over-claiming from a 2-year, 37-name slice.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey, W. K., & West, K. D. (1987). *A simple, positive
  semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix.*
  Econometrica, 55(3), 703–708.
- **Reporting-lag discipline.** Fiscal year y predicts calendar-year y+1 returns — the
  same conservative lag used in Studies 231, 244, and 522 on this desk.

## Related desk studies

- **[Study 244 — Asset-Growth](../244-asset-growth/)**: the total-asset-growth parent
  anomaly (Cooper–Gulen–Schill); None/Mirage on a survivor S&P 500 panel.
- **[Study 231 — Sloan-Accruals](../231-sloan-accruals/)**: the working-capital accrual
  anomaly (inventory is an accrual component).
- **[Study 522 — Percent-Operating-Accruals](../522-percent-operating-accruals/)**: a
  scaled cousin on the accrual side of the same balance-sheet-bloat family.
