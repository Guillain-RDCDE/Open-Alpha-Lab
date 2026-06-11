# References & literature map — Study 51 (Blue-Chip)

## The effect and its source

- **Novy-Marx, R. (2013).** *The Other Side of Value: The Gross Profitability Premium.* Journal of
  Financial Economics 108(1) — gross profit / assets as the cleanest quality signal; high-GP firms
  out-earn low-GP firms, and the premium is roughly as strong as value.
- **Asness, C., Frazzini, A., & Pedersen, L. (2019).** *Quality Minus Junk.* Review of Accounting
  Studies — the broad "quality" factor (profitability, growth, safety, payout) and its premium.
- **Vendor entry** — the profitability/quality family on
  [paperswithbacktest](https://github.com/paperswithbacktest/awesome-systematic-trading); backlog:
  [`docs/pwb_strategies_inventory.md`](../../../docs/pwb_strategies_inventory.md).

## Why our sample under-powers it

- **Hou, K., Xue, C., & Zhang, L. (2020).** *Replicating Anomalies* — profitability survives replication
  better than most, but magnitude depends on universe breadth and sample; XBRL (~2007+) is short.
- The strong pre-2007 quality decades pre-date machine-readable SEC fundamentals, so a free EDGAR pull
  cannot see them — the honest limit stated in the verdict.

## Data

- **SEC EDGAR** — `us-gaap:GrossProfit` and `us-gaap:Assets` (10-K, fiscal-year), via
  `data.sec.gov/api/xbrl/companyconcept`. **Yahoo! Finance** — annual total returns. Universe: current
  S&P 500 members (survivorship-biased, large-cap, ~2007+). The offline synthetic panel injects a known
  quality premium (and a null).

*Contrast with the inverted [53 Jackpot](../../53-jackpot/) / [54 Static](../../54-static/): quality at
least points the right way on large caps. Companion fundamental study: [52 Smoke-Screen](../../52-smoke-screen/)
(accruals).*
