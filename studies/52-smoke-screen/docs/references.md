# References & literature map — Study 52 (Smoke-Screen)

## The effect and its source

- **Sloan, R. (1996).** *Do Stock Prices Fully Reflect Information in Accruals and Cash Flows about
  Future Earnings?* The Accounting Review 71(3) — the accruals anomaly: high-accruals firms underperform
  because the market over-weights the (less persistent) accrual component of earnings.
- **Richardson, S., Sloan, R., Soliman, M., & Tuna, I. (2005).** *Accrual Reliability, Earnings
  Persistence and Stock Prices.* Journal of Accounting and Economics — refinements of the measure.
- **Vendor entry** — the earnings-quality / accruals family on
  [paperswithbacktest](https://github.com/paperswithbacktest/awesome-systematic-trading); backlog:
  [`docs/pwb_strategies_inventory.md`](../../../docs/pwb_strategies_inventory.md).

## On the post-publication fade and tradability

- **Green, J., Hand, J., & Soliman, M. (2011).** *Going, Going, Gone? The Apparent Demise of the Accruals
  Anomaly.* Management Science — the effect weakened markedly after ~2003 as it was arbitraged; our
  short XBRL window can't see that, hence the FRAGILE tradability stamp.

## Data

- **SEC EDGAR** — `us-gaap:NetIncomeLoss`, `us-gaap:NetCashProvidedByUsedInOperatingActivities`,
  `us-gaap:Assets` (10-K, fiscal-year). **Yahoo! Finance** — annual total returns. Universe: current
  S&P 500 members (survivorship-biased, large-cap, ~2007+). The offline synthetic panel injects a known
  accruals premium (and a null).

*Companion fundamental study: [51 Blue-Chip](../../51-blue-chip/) (quality / gross profitability) — both
read the same shared EDGAR pull; accruals replicates more strongly here.*
