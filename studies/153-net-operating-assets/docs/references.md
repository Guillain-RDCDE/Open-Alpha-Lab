# References & literature map — Study 153 (Net-Operating-Assets)

## The foundational paper

- **Hirshleifer, D., Hou, K., Teoh, S. H., & Zhang, Y. (2004).** *Do investors
  overvalue firms with bloated balance sheets?* Journal of Accounting and Economics,
  38, 297–331. The original NOA anomaly paper. Constructs NOA as (Operating Assets −
  Operating Liabilities) / lagged Total Assets and documents a robust negative
  cross-sectional predictability for returns, especially among stocks with high analyst
  following (arguing the market over-extrapolates accounting-based past performance
  signals). On the original all-stock sample: top-minus-bottom-NOA quintile earns
  roughly −12%/yr on a value-weighted basis (i.e., high bloat underperforms by
  ~12%/yr). The effect is larger among small caps.

## Replication and related work

- **Fairfield, P. M., Whisenant, J. S., & Yohn, T. L. (2003).** *Accrual components,
  earnings growth and the implication for future profitability and stock returns.*
  Working paper (later published in The Accounting Review). Documents that total accruals
  (growth in net operating assets) negatively predict future profitability and returns
  — closely related to the NOA anomaly; both capture balance-sheet bloat.

- **Richardson, S. A., Sloan, R. G., Soliman, M. T., & Tuna, I. (2005).** *Accrual
  reliability, earnings persistence and stock prices.* Journal of Accounting and
  Economics, 39(3), 437–485. Decomposes accruals into working capital and long-term
  components; finds the NOA (long-term accrual) component also strongly predicts
  returns, consistent with Hirshleifer et al.

- **Sloan, R. G. (1996).** *Do stock prices fully reflect information in accruals and
  cash flows about future earnings?* The Accounting Review, 71(3), 289–315. The seminal
  accruals-anomaly paper. Shows that high total accruals (accounting earnings > cash
  earnings) predict low future returns. NOA is a balance-sheet-level generalisation:
  not just working-capital accruals but the entire operating-asset surplus.

- **Fama, E. F., & French, K. R. (2006).** *Profitability, investment and average
  returns.* Journal of Financial Economics, 82(3), 491–518. Finds that controlling for
  book-to-market and profitability, high asset growth (related to high NOA) predicts low
  returns — corroborating the NOA signal from a different angle.

- **Cooper, M. J., Gulen, H., & Schill, M. J. (2008).** *Asset growth and the cross
  section of stock returns.* Journal of Finance, 63(4), 1609–1651. Documents a broad
  asset-growth effect (high growth → low returns) that generalises the NOA signal;
  finds the effect survives controls for size, book-to-market, and momentum.

## Why the effect exists (the economic mechanism debate)

- **Hirshleifer et al. (2004)** argue for *investor over-extrapolation*: investors
  see past earnings growth (supported by asset expansion) and extrapolate it forward,
  overvaluing high-NOA firms. Returns mean-revert as the expansion produces disappointing
  earnings.

- The **rational-risk alternative**: high NOA might proxy for investment risk (firms that
  invest aggressively bear more uncertainty). Fama & French (2006) and Cooper et al.
  (2008) investigate but cannot fully explain the effect on a rational basis. The weight
  of evidence tilts toward mispricing.

- **Limits to arbitrage**: the effect is larger among smaller, more illiquid stocks
  (consistent with mispricing persisting where arbitrage is costly). The survivorship-biased
  S&P 500 panel here is a *poor* testing ground precisely because these are large, liquid,
  heavily covered names where the over-extrapolation channel is least likely to operate.

## Survivorship bias and data limitations

- The EDGAR cache used here covers *current* S&P 500 members only, projected backwards.
  This excludes companies that were removed from the index (for distress, bankruptcy, or
  delistment) — precisely the high-NOA firms that would drive the anomaly in the correct
  direction. See **Kothari, S. P., Sabino, J., & Zach, T. (2005).** *Implications of
  survival and data trimming for tests of market efficiency.* Journal of Accounting and
  Economics, 39(1), 129–161.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.summary`](../net_operating_assets/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Reporting lag discipline.** Fundamentals from fiscal year y predict returns in
  calendar year y+1 — the same conservative lag used in Studies 52, 65, and 121 on
  this desk.

## Related desk studies

- **[Study 52 — Smoke-Screen](../../52-smoke-screen/)**: accruals anomaly (Sloan 1996),
  same EDGAR panel — the working-capital version of balance-sheet bloat.
- **[Study 65 — Scorecard](../../65-scorecard/)**: Piotroski F-score on the same panel.
- **[Study 121 — Magic-Formula](../../121-magic-formula/)**: Greenblatt's quality+value
  rank on the same panel; NOA-type over-investment captured differently via ROC.
