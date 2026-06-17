# References & literature map — Study 244 (Asset-Growth)

## The foundational paper

- **Cooper, M. J., Gulen, H., & Schill, M. J. (2008).** *Asset growth and the cross
  section of stock returns.* Journal of Finance, 63(4), 1609–1651. The original
  asset-growth anomaly paper. Defines asset-growth as YoY total-asset change divided
  by prior-year total assets. Documents a robust negative cross-sectional predictability
  — high asset-growth firms earn roughly −8%/yr relative to low-growth firms — across
  all NYSE/AMEX/NASDAQ stocks 1968–2003. The effect is pervasive, surviving controls
  for size, book-to-market, momentum, and accruals.

## Replication and related work

- **Titman, S., Wei, K. C. J., & Xie, F. (2004).** *Capital investments and stock
  returns.* Journal of Financial and Quantitative Analysis, 39(4), 677–700. Documents
  that firms with unusually high capital expenditures underperform — a related "over-
  investment" anomaly that anticipates Cooper et al.'s broader asset-growth measure.

- **Fairfield, P. M., Whisenant, J. S., & Yohn, T. L. (2003).** *Accrual components,
  earnings growth and the implication for future profitability and stock returns.*
  Working paper. Shows that growth in net operating assets (total asset expansion)
  negatively predicts future profitability and returns — closely related to the
  Cooper et al. asset-growth measure.

- **Hirshleifer, D., Hou, K., Teoh, S. H., & Zhang, Y. (2004).** *Do investors
  overvalue firms with bloated balance sheets?* Journal of Accounting and Economics,
  38, 297–331. The NOA anomaly — a related signal focusing on the ratio of operating
  assets to prior assets. Tested in [Study 153 — Net-Operating-Assets](../../153-net-operating-assets/).

- **Fama, E. F., & French, K. R. (2006).** *Profitability, investment and average
  returns.* Journal of Financial Economics, 82(3), 491–518. Investment (asset growth)
  enters the Fama-French factor framework: high investment → low expected returns.
  The investment factor (CMA — Conservative Minus Aggressive) in the Fama-French
  five-factor model (2015) directly operationalises this finding.

- **Fama, E. F., & French, K. R. (2015).** *A five-factor asset pricing model.*
  Journal of Financial Economics, 116(1), 1–22. Introduces CMA (Conservative Minus
  Aggressive investment). Firms with low investment (conservative) earn higher returns.
  Asset-growth is the long-side benchmark.

- **Gray, W. R., & Vogel, J. R. (2016).** *Quantitative Momentum.* Wiley. Practical
  discussion of the investment factor and asset-growth anomaly in a multi-factor
  framework.

## Why the effect exists (the economic mechanism debate)

- **Over-investment / agency costs.** Managers invest beyond the value-maximising level
  when free cash flow is plentiful (Jensen 1986). Markets are slow to discount the
  destruction of capital, producing temporary overvaluation of high-growth firms.

- **Rational risk pricing.** Fama & French argue the investment factor captures rational
  variation in discount rates — high-investment firms are in expansion phases when
  discount rates are low, so expected returns are also low. This is the q-theory
  interpretation.

- **Investor over-extrapolation.** Similar to the NOA mechanism: investors see past
  asset growth, extrapolate future profitability, and overpay. Returns disappoint as
  growth mean-reverts.

- **Limits to arbitrage.** The effect is strongest among small, illiquid stocks where
  short-selling is costly. On large liquid S&P 500 names, any mispricing is quickly
  arbitraged away — consistent with the absent effect on this survivor panel.

## Survivorship bias and data limitations

- The EDGAR cache used here covers *current* S&P 500 members only, projected backwards.
  This excludes companies that failed, were acquired, or were removed from the index —
  including high-growth firms that subsequently failed (the very firms that would drive
  the anomaly in the predicted direction). See **Kothari, S. P., Sabino, J., & Zach, T.
  (2005).** *Implications of survival and data trimming for tests of market efficiency.*
  Journal of Accounting and Economics, 39(1), 129–161.

- Additionally, the survivor S&P 500 panel includes successful fast-growing technology
  companies (Apple, Amazon, Nvidia, Google) that dominated the 2010s bull market.
  These are in the high-AG quintile and earned extraordinary returns — the opposite of
  what the anomaly predicts. On a balanced (non-biased) universe, these would be
  offset by high-growth failures.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica).
- **Reporting lag discipline.** Fundamentals from fiscal year y predict returns in
  calendar year y+1 — the same conservative lag used in Studies 52, 65, 121, and 153
  on this desk.

## Related desk studies

- **[Study 153 — Net-Operating-Assets](../../153-net-operating-assets/)**: NOA anomaly
  (Hirshleifer et al. 2004) — a scaled cousin of asset-growth; Weak/Fragile on the
  survivor panel.
- **[Study 138 — Random-Forest](../../138-random-forest/)**: ML approach to the same
  panel; None/Mirage.
- **[Study 121 — Magic-Formula](../../121-magic-formula/)**: Greenblatt quality+value
  rank, same EDGAR panel; over-investment captured via ROC.
