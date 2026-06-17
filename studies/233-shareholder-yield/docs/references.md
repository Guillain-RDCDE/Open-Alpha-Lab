# References & literature map — Study 233 (Shareholder-Yield)

## The claim and its academic basis

- **Faber, M. T. (2007).** *A Quantitative Approach to Tactical Asset Allocation.* Journal of Wealth
  Management — popularised "shareholder yield" (dividends + buybacks) as a broader capital-return
  factor; showed composite yield outperforms dividend yield alone on historical U.S. data.
- **Boudoukh, J., Michaely, R., Richardson, M., & Roberts, M. R. (2007).** *On the Importance of
  Measuring Payout Yield: Implications for Empirical Asset Pricing.* Journal of Finance 62(2) —
  total payout yield (dividends + net buybacks) predicts market returns better than dividend yield;
  the signal is stronger in the cross-section than in the aggregate.
- **Grullon, G., & Michaely, R. (2002).** *Dividends, Share Repurchases, and the Substitution
  Hypothesis.* Journal of Finance 57(4) — documents the substitution of repurchases for dividends,
  motivating the composite yield definition.

## Net share issuance literature (the buyback-yield building block)

- **Pontiff, J., & Woodgate, A. (2008).** *Share Issuance and Cross-Sectional Returns.* Journal of
  Finance 63(2) — net share issuance negatively predicts returns. This is the mirror image of the
  buyback-yield leg.
- **Daniel, K., & Titman, J. (2006).** *Market Reactions to Tangible and Intangible Information.*
  Journal of Finance — composite issuance (dilution from all sources) predicts returns.
- **Open-Alpha-Lab**: [64 Share-Shuffle](../../64-share-shuffle/) — the net-issuance building block
  reused here; its long-buyback/short-issuer hedge also inverts on large caps (−2.8%/yr, t −1.3).

## Why the signal inverts on tradable large caps

- **Hou, K., Xue, C., & Zhang, L. (2020).** *Replicating Anomalies.* Review of Financial Studies 33(5)
  — issuance/financing anomalies concentrate in micro-caps and fade with value-weighting and NYSE
  breakpoints.
- **McLean, R. D., & Pontiff, J. (2016).** *Does Academic Research Destroy Stock Return Predictability?*
  Journal of Finance 71(1) — post-publication decay of anomalies; shareholder yield has been traded
  and likely arbitraged in large-cap space since publication.
- **Open-Alpha-Lab** kin: [44 Growth-Spurt](../../44-growth-spurt/) (asset growth inverts on large
  caps), [52 Smoke-Screen](../../52-smoke-screen/) (accruals — *does* replicate), [64 Share-Shuffle](
  ../../64-share-shuffle/) (pure buyback/issuance — also inverts).

## Data

- **SEC EDGAR** — `us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding` (10-K, FY), year-on-year
  change = net buyback yield. Universe: current S&P 500 members (survivorship opt-in, large-cap, XBRL
  era ~2008+). **Yahoo! Finance** — annual total returns.
- The offline machinery proof uses a synthetic panel with a planted yield premium (and a null world)
  inside [`shareholder_yield/data.py`](../shareholder_yield/data.py).
