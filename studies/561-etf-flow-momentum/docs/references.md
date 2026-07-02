# References & literature map — Study 561 (ETF-Flow-Momentum)

## The claim, at full strength (flows keep winning)

- **Ben-Rephael, Kandel & Wohl (2012)**, *"Measuring Investor Sentiment with Mutual Fund Flows."*
  *Journal of Financial Economics* 104(2). Aggregate fund flows are strongly persistent and
  positively related to contemporaneous and near-term returns — the empirical backbone of the
  "flows predict continued performance" view this study plants as `flow_alpha > 0`.
- **Clifford, Fulkerson, Jordan & Waldman (2014)**, *"Do Investors Care about Risk? Evidence from
  Mutual Fund Flows."* Flow-performance sensitivity: money chases recent winners, and the chased
  funds/sectors often keep leading over short horizons.
- **Dannhauser (2017)**, *"The Impact of Innovation: Evidence from Corporate Bond ETFs."* *JFE*
  125(3). ETF creation/redemption activity and flows move underlying prices — the mechanism by
  which inflow pressure could sustain a sector's outperformance.

## The counter-claim (chasing flows is a trap — reversal / crowding)

- **Frazzini & Lamont (2008)**, *"Dumb Money: Mutual Fund Flows and the Cross-Section of Stock
  Returns."* *JFE* 88(2). The seminal reversal result: the stocks/sectors retail money floods into
  subsequently *underperform* — flows are a *negative* predictor at longer horizons. This is the
  study's `flow_alpha < 0` world (the trap).
- **Ben-David, Franzoni & Moussawi (2018)**, *"Do ETFs Increase Volatility?"* *Journal of Finance*
  73(6). ETF ownership and flow-driven demand shocks propagate non-fundamental price pressure that
  **mean-reverts** — heavy-inflow ETFs are exactly the over-extended ones set to give it back.
- **Brown, Davies & Ringgenberg (2021)**, *"ETF Arbitrage, Non-Fundamental Demand, and Return
  Predictability."* *Review of Finance*. ETF flows driven by non-fundamental demand *negatively*
  predict future returns — direct evidence for the reversal side of the sign dispute.

## Why the sign is the whole question

The two literatures disagree on the *sign* of the flow→forward-return relation at the horizon a
retail trader could act on. That unresolved sign is precisely why a study with no real tape must
read `WEAK`, not `REAL`: the claim has genuine academic support, but it competes head-on with an
equally well-documented reversal, and only a clean real flow panel could adjudicate.

## Why this study is synthetic-only

Honest ETF creation-unit flow requires a **daily shares-outstanding history × NAV**. That series is
a paid vendor field (Bloomberg / FactSet / issuer creation-redemption files); yfinance exposes only
a single stale `sharesOutstanding` scalar per ETF, so a free `price × Δshares` proxy would inject
staleness and look-ahead. The desk's rubric caps a synthetic-only study at `WEAK`/`NONE` and names
the data-availability limit on the SIGNAL axis — the same convention as the
[lego-returns](../../273-lego-returns/), [whisky-cask](../../275-whisky-cask/) and
[sneaker-resale](../../276-sneaker-resale/) studies.

## Neighbours on this bench (the dedup map)

- **[Study 378 — ETF-NAV-Premium](../../378-etf-nav-premium/)** — the ETF *price-vs-NAV* premium,
  an arbitrage/mispricing signal. Study 561 is about *flows* (creation-unit inflows) predicting
  *sector returns*, not the price-NAV gap.
- **[Study 379 — ETF-Lead-Lag](../../379-etf-lead-lag/)** — cross-ETF lead-lag in returns. Study
  561's predictor is the *flow* variable, not lagged returns.
- **[Study 518 — Time-Series-Momentum](../../518-time-series-momentum/)** /
  **[Study 507 — Cross-Sectional-Momentum](../../507-cross-sectional-momentum/)** — *return*-based
  momentum. Study 561 sorts on *flow*, a different (and separately disputed) predictor, and asks
  whether flow-chasing persists or reverses.
- **[Study 335 — Buzz-Sentiment-ETF](../../335-buzz-sentiment-etf/)** — a sentiment/attention
  signal on ETFs. Flows overlap attention but are a distinct, balance-sheet-observable quantity.

## Shared method

- **One-sample *t*** on the monthly long-short spread series — the headline inference bar.
- **Fama & MacBeth (1973)** cross-sectional logic — the pooled slope's *t* is built from per-month
  slope estimates (months as the independent units), robust to within-month cross-ETF correlation.
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null: shuffle the
  flow labels against forward returns *within each month* and read the mean spread's tail
  probability.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a robust
  *t* ≥ 2 on a **real** tape for `REAL`; synthetic-only caps at `WEAK`/`NONE`), the seed-robust
  synthetic control (≥ 20 seeds), one execution lag, and costs one-way × NAV with shorts paying
  borrow.
