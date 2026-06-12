# References & literature map — Study 65 (Scorecard)

## The screen and its source

- **Piotroski, J. (2000).** *Value Investing: The Use of Historical Financial Statement Information to
  Separate Winners from Losers.* Journal of Accounting Research 38 — the nine-point F-score, built and
  tested **within high book-to-market (value) firms**, concentrated in small, neglected names.
- **Piotroski, J., & So, E. (2012).** *Identifying Expectation Errors in Value/Glamour Strategies.*
  Review of Financial Studies — the F-score works by correcting mispriced expectations, strongest where
  coverage is thin.

## Why it fades on large caps

- **Fama, E., & French, K. (2008).** *Dissecting Anomalies.* Journal of Finance — many accounting
  anomalies are driven by micro-caps and weaken in large-cap, value-weighted tests.
- **Hou, K., Xue, C., & Zhang, L. (2020).** *Replicating Anomalies* — fundamental screens shrink under
  NYSE breakpoints and value-weighting.
- **Open-Alpha-Lab** kin: [51 Blue-Chip](../../51-blue-chip/) (quality/profitability), [52
  Smoke-Screen](../../52-smoke-screen/) (accruals, which *does* replicate), [64
  Share-Shuffle](../../64-share-shuffle/) (net issuance, also inverted on large caps).

## Data

- **SEC EDGAR** — the nine `us-gaap` concepts behind the F-score (`NetIncomeLoss`,
  `NetCashProvidedByUsedInOperatingActivities`, `Assets`, `GrossProfit`, `Revenues`,
  `WeightedAverageNumberOfDilutedSharesOutstanding`, `LongTermDebtNoncurrent`, `AssetsCurrent`,
  `LiabilitiesCurrent`), 10-K / FY. **Yahoo! Finance** — annual total returns. Universe: current S&P 500
  members (survivorship opt-in, large-cap, ~2009+). The offline synthetic panel injects a known F-score
  premium (and a null).

*A fundamental-screen companion to [51 Blue-Chip](../../51-blue-chip/) and [52
Smoke-Screen](../../52-smoke-screen/); shares the inverted-on-large-caps pattern of [64
Share-Shuffle](../../64-share-shuffle/) and [44 Growth-Spurt](../../44-growth-spurt/).*
