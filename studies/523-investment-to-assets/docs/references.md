# References & literature map — Study 523 (Investment-To-Assets)

## The foundational papers

- **Titman, S., Wei, K. C. J., & Xie, F. (2004).** *Capital investments and stock returns.*
  Journal of Financial and Quantitative Analysis, 39(4), 677–700. The canonical
  capex-channel paper. Sorts firms on abnormal capital investment (capex scaled by a base
  of assets / sales / prior capex) and documents that the heaviest investors subsequently
  *underperform* — roughly −7%/yr for the high-minus-low quintile — strongest among firms
  with the greatest discretion and weakest governance. This is the over-investment anomaly
  in its purest capital-expenditure form, and the signal this study replicates.

- **Cooper, M. J., Gulen, H., & Schill, M. J. (2008).** *Asset growth and the cross section
  of stock returns.* Journal of Finance, 63(4), 1609–1651. The broader total-asset-growth
  generalisation of the investment anomaly. Tested separately on this desk in
  [Study 244 — Asset-Growth](../../244-asset-growth/). Study 523 isolates the *capex/PP&E*
  channel rather than the total-balance-sheet channel.

## The factor-level packaging

- **Fama, E. F., & French, K. R. (2015).** *A five-factor asset pricing model.* Journal of
  Financial Economics, 116(1), 1–22. Introduces **CMA** (Conservative Minus Aggressive
  investment). Low-investment ("conservative") firms earn higher returns than
  high-investment ("aggressive") firms — the factor-level operationalisation of the
  investment anomaly.

- **Hou, K., Xue, C., & Zhang, L. (2015).** *Digesting anomalies: an investment approach.*
  Review of Financial Studies, 28(3), 650–705. The **q-factor** model, whose investment
  factor (I/A — investment-to-assets, change in total assets over lagged assets) is the
  direct quantitative cousin of the signal here. The q-theory reading: high-investment
  firms invest precisely when discount rates (and hence expected returns) are low.

- **Fama, E. F., & French, K. R. (2006).** *Profitability, investment and average returns.*
  Journal of Financial Economics, 82(3), 491–518. Embeds investment (asset growth) in the
  expected-return framework: high investment → low expected returns.

## Why the effect exists (mechanism debate)

- **Over-investment / agency costs.** Jensen (1986) free-cash-flow agency theory — managers
  invest beyond the value-maximising level when capital is cheap; the market is slow to
  discount the capital destruction, producing temporary overvaluation of heavy investors.

- **Rational q-theory.** Hou-Xue-Zhang and Fama-French argue the investment factor captures
  rational variation in discount rates rather than mispricing: firms invest more when the
  cost of capital is low, mechanically lowering expected returns.

- **Limits to arbitrage.** Titman-Wei-Xie find the effect concentrated among firms with the
  most managerial discretion and is weaker for large, well-governed, heavily-covered names —
  consistent with the absent effect on this large-cap survivor basket.

## Survivorship bias and data limitations

- **Kothari, S. P., Sabino, J., & Zach, T. (2005).** *Implications of survival and data
  trimming for tests of market efficiency.* Journal of Accounting and Economics, 39(1),
  129–161. The basket here covers only large-cap names still trading in 2026; capex-heavy
  over-investors that subsequently failed are absent, biasing any real-tape result upward
  and, on a survivor basket, often reversing the predicted sign (the high-IA survivors are
  productive expanders like semiconductor and datacentre builders).

- **Data depth.** Yahoo Finance statement endpoints expose only ~5 fiscal years; this study
  instead pulls **SEC EDGAR companyfacts** (`PaymentsToAcquirePropertyPlantAndEquipment`
  and `Assets`) to obtain ~13 years of annual fundamentals, yielding 16 stampable
  cross-sectional sort years. Some basket names (financials, telecoms) do not report the
  standard capex GAAP tag and drop out, leaving ~27 names/year.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) — the
  one-sample inference on the short annual hedge series.
- **Reporting-lag discipline.** IA from fiscal year y drives a 12-month return beginning a
  conservative 4 months after the fiscal-year-end (when the 10-K is public), entered one
  trading day later — the same single-execution-lag discipline used across the desk's
  fundamental sorts (Studies 52, 65, 121, 153, 244).

## Related desk studies

- **[Study 244 — Asset-Growth](../../244-asset-growth/)**: the total-asset-growth channel of
  the same anomaly (Cooper-Gulen-Schill) — None/Mirage on the survivor panel.
- **[Study 153 — Net-Operating-Assets](../../153-net-operating-assets/)**: the balance-sheet
  bloat cousin (Hirshleifer et al.) — Weak/Fragile.
- **[Study 121 — Magic-Formula](../../121-magic-formula/)**: quality+value rank that captures
  return-on-capital, the flip side of over-investment.
