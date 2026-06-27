# References & literature map — Study 521 (Cash-Based Operating Profitability)

## The primary claim under test

- **Ball, R., Gerakos, J., Linnainmaa, J. T. & Nikolaev, V. (2016).** "Accruals, cash
  flows, and operating profitability in the cross section of stock returns." *Journal of
  Financial Economics*, 121(1), 28–45. The founding paper. They decompose operating
  profitability into a **cash** component and an **accrual** component, and show that
  *cash-based* operating profitability predicts the cross-section of returns substantially
  better than the accrual-laden gross/operating profitability of Novy-Marx (2013). A
  long-short on cash-based OP earns a higher, more robust premium and subsumes both the
  gross-profitability and the accruals anomalies. The mechanism: accruals are the noisy,
  reversal-prone part of earnings; stripping them out isolates the durable, value-relevant
  signal. This is the signal we replicate.

## The accrual-laden sibling it is contrasted against

- **Novy-Marx, R. (2013).** "The Other Side of Value: The Gross Profitability Premium."
  *Journal of Financial Economics*, 108(1), 1–28. GrossProfit / Assets predicts the
  cross-section as reliably as book-to-market. This is the accrual-laden measure that BGLN
  argue is dominated by their cash version — replicated as Study 122 on this desk and used
  here as the head-to-head benchmark.

## The accruals anomaly — why stripping accruals helps

- **Sloan, R. G. (1996).** "Do Stock Prices Fully Reflect Information in Accruals and Cash
  Flows About Future Earnings?" *The Accounting Review*, 71(3), 289–315. The original
  accruals anomaly: firms with high accruals (relative to cash earnings) underperform.
  Accruals revert; cash earnings persist. BGLN's cash-based profitability is a direct
  descendant — it keeps the persistent cash part and discards the reverting accrual part.
- **Fama, E. F. & French, K. R. (2006).** "Profitability, Investment, and Average Returns."
  *Journal of Financial Economics*, 82(3), 491–518. The valuation backbone: holding
  book-to-market fixed, more profitable firms should earn higher expected returns. Cash-OP
  is a cleaner empirical proxy for the "profitability" in that identity.
- **Fama, E. F. & French, K. R. (2015).** "A Five-Factor Asset Pricing Model." *Journal of
  Financial Economics*, 116(1), 1–22. Adds the RMW (robust-minus-weak) profitability factor;
  BGLN show a cash-based construction of RMW is stronger than the operating-profit version.

## Subsequent evidence, replication, and attenuation

- **Hou, K., Xue, C. & Zhang, L. (2020).** "Replicating Anomalies." *Review of Financial
  Studies*, 33(5), 2019–2133. Cash-based operating profitability is among the more robust
  survivors of their ~400-anomaly replication, but magnitudes attenuate out of the original
  sample.
- **McLean, R. D. & Pontiff, J. (2016).** "Does Academic Research Destroy Stock Return
  Predictability?" *Journal of Finance*, 71(1), 5–32. Cross-sectional predictors weaken by
  ~32% post-publication; the cash-OP premium, published in 2016, is exposed to the same
  decay over our 2022–2024 window.

## Survivorship bias and data limitations

- **Shumway, T. & Warther, V. A. (1999).** "The Delisting Bias in CRSP's Nasdaq Data."
  *Journal of Finance*, 54(6), 2361–2379. Delistings are not random; excluding failed firms
  (natural short candidates) biases factor premia upward. Our survivor basket inherits this.
- **Linnainmaa, J. T. & Roberts, M. R. (2018).** "The History of the Cross-Section of Stock
  Returns." *Review of Financial Studies*, 31(7), 2606–2649. Many accounting anomalies are
  weaker out of the discovery sample; selection and look-back matter. (Linnainmaa is a
  co-author of the BGLN paper — a useful caution from the same author.)

## Trading costs and method lineage

- **Novy-Marx, R. & Velikov, M. (2016).** "A Taxonomy of Anomalies and Their Trading Costs."
  *Review of Financial Studies*, 29(1), 104–147. Annual profitability sorts have modest
  turnover (~30–50%/yr at large-cap), so transaction costs are not the binding constraint —
  consistent with our 0.67%/yr cost charge.
- **Newey, W. K. & West, K. D. (1987).** "A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica*,
  55(3), 703–708. The HAC lineage behind the desk's annual-series inference (here the panel
  is too short for HAC, so we report the plain one-sample t and a permutation placebo).

## Related desk studies

- **[Study 122 — Gross-Profitability](../../122-gross-profitability)**: the accrual-laden
  Novy-Marx GP/A sort — the direct sibling BGLN's cash measure is meant to beat.
- **[Study 124 — Cash-Flow Yield](../../124-cash-flow-yield)**: a cash-based valuation
  signal — complementary cash dimension.
- **[Study 238 — Betting-Against-Beta](../../238-betting-against-beta)**: same survivor
  basket philosophy and long-short engine shape.
