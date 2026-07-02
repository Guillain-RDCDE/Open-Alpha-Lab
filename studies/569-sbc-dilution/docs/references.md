# References & literature map — Study 569 (SBC-Dilution)

## The claim, at full strength

- **Pontiff & Woodgate (2008)**, *"Share Issuance and Cross-Sectional Returns."* *Journal of
  Finance* 63(2). The canonical **net-share-issuance** factor: the year-over-year change in shares
  outstanding negatively predicts returns — diluters underperform, repurchasers outperform. SBC is
  a *mechanism* of dilution, so this is the parent factor Study 569's dilution leg proxies.
- **Daniel & Titman (2006)**, *"Market Reactions to Tangible and Intangible Information."*
  *Journal of Finance* 61(4). The **composite issuance** measure (the part of book-value growth not
  explained by earnings) predicts low returns — the intangible/hidden-cost framing that motivates
  scoring SBC intensity, not just the raw share count.
- **Bens, Nagar, Skinner & Wong (2003)**, *"Employee Stock Options, EPS Dilution, and Stock
  Repurchases."* *Journal of Accounting and Economics* 36. Firms buy back stock to offset
  SBC-driven dilution — the exact hidden-cost channel: SBC vesting dilutes, and the market may
  under-weight the offsetting repurchase cost.
- **Fitzgerald, Gray, Nguyen & Toohey (2010)** and related SBC-anomaly work: firms with high
  stock-based compensation relative to fundamentals have subsequently underperformed, consistent
  with SBC being an under-priced economic expense. The direct statement of the effect Study 569
  tests.

## The signal we build

- **Dilution score** = `z(SBC / revenue) + z(share-count growth)`, each z-scored across the basket
  within each formation year (higher = more dilutive). The SBC leg is `Stock Based Compensation`
  (cash-flow statement) / `Total Revenue`; the dilution leg is year-over-year growth in
  **split-adjusted** shares outstanding (`get_shares_full` corrected with the split history, so a
  split is not mistaken for issuance). yfinance exposes only a shallow ~4-year statement history,
  so the SBC leg is a recent snapshot — named on the SIGNAL axis and a hard cap on the stamp.

## Neighbours on this bench (the dedup map)

- **[Study 519 — Net-Share-Issuance](../../519-net-share-issuance/)** — the *pure* share-count
  factor (Pontiff-Woodgate). Study 569 is the **SBC-flavoured** cousin: it adds a stock-based-comp
  intensity leg to the dilution signal, testing the *hidden-cost* framing (equity paid to
  employees), not just the mechanical share count.
- **[Study 368 — Buyback-Drift](../../368-buyback-drift/)** — times discrete repurchase
  *announcements*; Study 569 is a *realised* annual cross-sectional sort on the dilution level, not
  an event study.
- **[Study 540 — Distress-Risk-Anomaly](../../540-distress-risk-anomaly/)** — another
  accounting-anomaly-on-a-survivor-basket study that **inverts** on the same kind of tape; the
  shared moral is that current-membership panels can flip a real cross-sectional effect.

## Shared method

- **One-sample *t* against zero** for the annual long-short mean — the honest test on a handful of
  years is whether it clears |*t*| = 2 *at all*, and with what sign.
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null: permute
  which name carries which dilution score within each year and read how often a shuffled book beats
  the real sort.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a real-tape
  *t* ≥ 2 plus a placebo null and seed-robustness), the explicit survivorship caveat named on the
  Signal axis, one execution lag, costs one-way × NAV with shorts paying borrow, and the
  synthetic-only / survivor-only cap (never `REAL`).
