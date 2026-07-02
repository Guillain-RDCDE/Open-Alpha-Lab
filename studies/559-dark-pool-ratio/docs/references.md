# References & literature map — Study 559 (Dark-Pool-Ratio)

## The claim, at full strength

- **The retail "dark pool index" folklore.** A cottage industry of trading blogs and vendors sells
  the idea that a rising **dark-pool ratio** (off-exchange share of volume) is *smart money quietly
  accumulating*, and that the stock drifts up afterwards. This study takes that claim at face value
  and tests whether the venue-of-execution predicts forward returns.
- **Zhu (2014)**, *"Do Dark Pools Harm Price Discovery?"* *Review of Financial Studies* 27(3). The
  key theory result: dark pools skim the *less* information-sensitive orders, so — contrary to the
  folklore — dark trading can *concentrate* informed traders on the lit exchange and improve lit
  price discovery. A caution that "more dark = more informed accumulation" is not obvious.
- **Comerton-Forde & Putniņš (2015)**, *"Dark Trading and Price Discovery."* *Journal of Financial
  Economics* 118(1). Empirically, *low* levels of dark trading are benign or beneficial for price
  discovery, but *high* levels harm it — the effect of dark share on information is **non-monotone**,
  undercutting a simple "high DPR = bullish" reading.
- **Buti, Rindi & Werner (2011/2017)**, *"Diving into Dark Pools."* Dark-pool activity rises with
  stock liquidity, market depth and tighter spreads — i.e. dark share is **correlated with
  size/liquidity**, the confound this study's synthetic panel plants and controls for.
- **Ye (2011)**, *"A Glimpse into the Dark: Price Formation, Transaction Cost and Market Share in the
  Crossing Network."* Informed traders' venue choice is subtle; the direction of any DPR→return
  relation is not pinned down by theory.

## Why there is no free real tape (the data-availability limitation)

- **FINRA ATS Transparency Data** publishes ATS ("dark pool") share volume by security, but only
  **weekly**, aggregated, and released on a **two-to-four-week lag** — not a clean point-in-time
  daily panel.
- The large **off-exchange, non-ATS** flow (wholesaler internalisation / single-dealer platforms,
  reported to the TRF) is *not* in the FINRA ATS file, so "off-exchange %" cannot be fully
  reconstructed from the free data.
- Vendors who sell a clean daily DPR (the feeds behind retail "dark pool" charts) are paywalled.

So a per-name daily DPR joined to forward returns is not buildable from a free, no-key retail stack.
This study is **synthetic-only** and the gap is named on the SIGNAL axis; it caps the verdict below
`REAL` (which requires a robust *t* ≥ 2 on a real tape).

## Neighbours on this bench (the dedup map)

- **[Study 376 — MOC-Imbalance](../../376-moc-imbalance/)** — the *market-on-close order imbalance*
  (buy vs sell pressure at the close). Study 559 is the **venue-of-execution** signal (where volume
  prints, lit vs dark), not the close-auction imbalance.
- **[Study 418 — Money-Flow-Index](../../418-money-flow-index/)** /
  **[Study 419 — Chaikin-Money-Flow](../../419-chaikin-money-flow/)** — price×volume *oscillators*
  that infer "flow" from price and volume on the lit tape. Study 559 is about *off-exchange* share,
  a genuinely different (and here unobservable-for-free) input.
- **[Study 540 — Distress-Risk-Anomaly](../../540-distress-risk-anomaly/)** — shares the
  cross-sectional-sort + IC + placebo + seed-robust positive-control machinery reused here.

## Shared method

- **Information coefficient (IC)** — the Spearman rank correlation between a factor and forward
  return, the standard "does this rank stocks?" statistic; significance via the **Fisher z-transform**
  *t* = *z*·√(*n*−3).
- **Welch (1947)** — the unequal-variance two-sample *t* for the quintile long-short spread.
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null: shuffle the
  DPR labels against forward returns and read the IC's tail probability.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a robust *t* ≥ 2
  on a *real* tape for `REAL`; a synthetic control is a machinery proof, never market evidence), the
  survivorship/confound honesty rules, one execution lag, and costs one-way × NAV with shorts paying
  borrow.
