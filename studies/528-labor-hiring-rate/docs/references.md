# References & literature map — Study 528 (Labor-Hiring-Rate)

## The foundational papers

- **Belo, F., Lin, X., & Bazdresch, S. (2014).** *Labor hiring, investment, and stock return
  predictability in the cross section.* Journal of Political Economy, 122(1), 129–177. The
  canonical hiring-rate paper. Builds a neoclassical investment model in which labor is a
  quasi-fixed factor with convex adjustment (hiring/firing) costs, so the **hiring rate**
  behaves like an investment rate. Empirically, firms with high hiring rates earn
  significantly *lower* future returns — a hiring-rate factor that prices the cross-section
  alongside the standard investment factor. This is the labor analogue of the
  capital-investment anomaly and the signal this study replicates.

- **Titman, S., Wei, K. C. J., & Xie, F. (2004).** *Capital investments and stock returns.*
  Journal of Financial and Quantitative Analysis, 39(4), 677–700. The capital-expenditure
  cousin: heavy capital investors subsequently underperform. Tested separately on this desk
  in [Study 523 — Investment-To-Assets](../523-investment-to-assets/). Study 528 isolates the
  *labor* input channel (hiring) rather than the *capital* input channel (capex).

- **Cooper, M. J., Gulen, H., & Schill, M. J. (2008).** *Asset growth and the cross section
  of stock returns.* Journal of Finance, 63(4), 1609–1651. The broad total-asset-growth
  generalisation of the investment anomaly. Tested on this desk in
  [Study 244 — Asset-Growth](../244-asset-growth/). Hiring growth and asset growth are
  correlated inputs of the same expansion.

## The factor-level packaging and theory

- **Fama, E. F., & French, K. R. (2015).** *A five-factor asset pricing model.* Journal of
  Financial Economics, 116(1), 1–22. Introduces **CMA** (Conservative Minus Aggressive
  investment). Belo-Lin-Bazdresch show the hiring rate carries return-predictive content
  *incremental* to such investment factors.

- **Hou, K., Xue, C., & Zhang, L. (2015).** *Digesting anomalies: an investment approach.*
  Review of Financial Studies, 28(3), 650–705. The **q-factor** model whose investment factor
  is the direct quantitative cousin; the q-theory reading of the hiring result is that firms
  hire most when discount rates (and hence expected returns) are low.

- **Belo, F., & Lin, X. (2012).** *The inventory growth spread.* Review of Financial Studies,
  25(1), 278–313. A companion real-input anomaly (inventory) from the same research program,
  reinforcing the "input growth predicts low returns" mechanism.

## Why the effect exists (mechanism debate)

- **Adjustment-cost / q-theory.** Belo-Lin-Bazdresch's structural reading: high hiring is
  optimal investment in the labor stock when the cost of capital is low, mechanically lowering
  expected returns — a rational, not behavioral, channel.

- **Over-extrapolation / agency (Jensen 1986).** The behavioral reading shared with the
  investment anomaly: managers over-expand (over-hire) when capital is cheap and the market
  over-extrapolates the head-count growth, producing temporary overvaluation of aggressive
  hirers.

- **Limits to arbitrage.** The investment/hiring effects are concentrated in small, less
  liquid, less-covered names — consistent with the absent (here reversed) effect on this
  large-cap survivor basket of the most heavily-arbitraged stocks in the market.

## Survivorship bias and data limitations

- **Kothari, S. P., Sabino, J., & Zach, T. (2005).** *Implications of survival and data
  trimming for tests of market efficiency.* Journal of Accounting and Economics, 39(1),
  129–161. The basket here covers only large-cap names still trading in 2026; aggressive
  hirers that over-expanded and subsequently failed are absent, biasing any real-tape result
  upward and, on a survivor basket, often *reversing* the predicted sign (the high-hiring
  survivors are productive growth names — semiconductors, hyperscalers, managed care).

- **Employee-count data are not machine-readable.** Unlike capex or total assets, U.S. firms
  report full-time-employee counts as **10-K cover-page narrative text**, not as a numeric
  XBRL fact. We verified that SEC EDGAR `companyfacts`, `companyconcept`, and `frames` return
  `dei:EntityNumberOfEmployees` for only ~5–11 filers market-wide, and that yfinance exposes
  only a single current `fullTimeEmployees` snapshot with no history. There is therefore *no
  API* for a deep employee time series. This study uses a **curated panel of 10-K cover-page
  headcounts** (FY2013–FY2024, public reported facts), hardcoded for offline reproducibility,
  with the latest column sanity-anchored to the live yfinance snapshot. This caps the sample
  at 11 stampable sort years — a real limitation noted on the verdict.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) — the
  one-sample inference on the short annual hedge series.
- **Reporting-lag discipline.** The year-y hiring rate drives a 12-month return beginning a
  conservative 4 months after the fiscal-year-end (when the 10-K is public), entered one
  trading day later — the same single-execution-lag discipline used across the desk's
  fundamental sorts (Studies 52, 65, 121, 244, 523).

## Related desk studies

- **[Study 523 — Investment-To-Assets](../523-investment-to-assets/)**: the capital-investment
  (capex) cousin of the same anomaly family — None/Mirage on the survivor panel.
- **[Study 244 — Asset-Growth](../244-asset-growth/)**: the total-asset-growth channel
  (Cooper-Gulen-Schill) — None/Mirage on the survivor panel.
- **[Study 524 — Operating-Leverage](../524-operating-leverage/)**: the cost-structure cousin —
  labor and operating leverage are linked through the fixity of the wage bill.
