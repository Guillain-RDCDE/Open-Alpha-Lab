# References & literature map — Study 64 (Share-Shuffle)

## The effect and its source

- **Pontiff, J., & Woodgate, A. (2008).** *Share Issuance and Cross-Sectional Returns.* Journal of
  Finance 63(2) — net share issuance negatively predicts returns; issuers underperform, buybacks
  outperform.
- **Daniel, K., & Titman, J. (2006).** *Market Reactions to Tangible and Intangible Information.*
  Journal of Finance — the composite-issuance variant.
- **Vendor / factor family** — issuance / buyback signals; backlog:
  [`docs/pwb_strategies_inventory.md`](../../../docs/pwb_strategies_inventory.md).

## Why it inverts on tradable large caps

- **Hou, K., Xue, C., & Zhang, L. (2020).** *Replicating Anomalies* — issuance/financing anomalies
  concentrate in micro-caps and weaken with value-weighting and NYSE breakpoints.
- **Open-Alpha-Lab** kin: [44 Growth-Spurt](../../44-growth-spurt/) (asset growth, the investment
  factor), [53 Jackpot](../../53-jackpot/) / [54 Static](../../54-static/) — all real small-cap effects
  that survivorship and the post-2009 growth regime invert on large-cap survivors.

## Data

- **SEC EDGAR** — `us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding` (10-K, FY), the share count
  whose year-on-year change is net issuance. **Yahoo! Finance** — annual total returns. Universe: current
  S&P 500 members (survivorship opt-in, large-cap, ~2010+). The offline synthetic panel injects a known
  issuance premium (and a null).

*A fundamental companion to [52 Smoke-Screen](../../52-smoke-screen/) (accruals, which *does* replicate)
and [51 Blue-Chip](../../51-blue-chip/) (quality); the inverted-on-large-caps pattern of [44](../../44-growth-spurt/).*
