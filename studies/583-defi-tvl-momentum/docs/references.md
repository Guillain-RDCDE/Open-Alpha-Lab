# References & literature map — Study 583 (DeFi-TVL-Momentum)

## The claim, at full strength

- **The on-chain folklore.** In DeFi, **total value locked (TVL)** — the dollar value of assets
  deposited in a protocol's smart contracts — is treated as the headline demand/health metric. The
  trader's version of the claim: rising TVL is *"smart money"* flowing in, a leading, public,
  real-time signal that the protocol's governance token will keep outperforming. The tradable
  expression is a cross-sectional **TVL-momentum** sort (long fastest inflows, short outflows).
- **DefiLlama** — the reference aggregator that popularised per-protocol TVL as the industry KPI;
  the source most retail dashboards and the folklore quote. Its free history is revised and
  re-categorised over time (the point-in-time problem this study names).

## Where the academic evidence actually sits

- **Momentum, cross-sectional** — Jegadeesh & Titman (1993), *"Returns to Buying Winners and
  Selling Losers."* *Journal of Finance* 48(1). The parent effect a TVL-momentum sort borrows its
  machinery from: rank a cross-section by a trailing signal, long the top / short the bottom.
- **Crypto momentum** — the price-momentum literature in crypto is mixed and fragile
  (short-horizon momentum, long-horizon reversal), which is why an *on-chain-flow* momentum signal
  is interesting but unproven. This desk's own crypto studies (below) find crypto anomalies mostly
  do not survive costs and survivorship.
- **Fund/ETF flows and returns** — the TradFi analogue of "flows predict returns" is itself weak
  and largely a demand-pressure/reversal story, not a durable alpha. The DeFi-TVL claim inherits
  that skepticism.

## The measurement problem (why this is synthetic-only)

- A faithful real test needs a **survivorship-free, point-in-time monthly panel** of per-protocol
  TVL joined to the same protocols' **token total returns**, with dead protocols (rugs, exploits,
  abandoned forks) still present. Free, no-key sources deliver a *revised, survivorship-selected*
  series. Per the desk rule for the [273 Lego-Returns](../../273-lego-returns/),
  [275 Whisky-Cask](../../275-whisky-cask/) and [276 Sneaker-Resale](../../276-sneaker-resale/)
  kind of study, a synthetic-only build **caps the Signal at WEAK** and states the data wall on the
  Signal axis.

## Neighbours on this bench (the dedup map)

- **[Study 133 — Crypto-Seasonality](../../133-crypto-seasonality/)**,
  **[Study 175 — Crypto-Weekend](../../175-crypto-weekend/)**,
  **[Study 210 — Crypto-Trend](../../210-crypto-trend/)**,
  **[Study 251 — Crypto-Reversal](../../251-crypto-reversal/)**,
  **[Study 325 — Crypto-Fear-Greed](../../325-crypto-fear-greed/)** — the desk's crypto folklore
  cluster. Those test *price*-derived crypto signals on a real tape; Study 583 tests an
  **on-chain fundamental flow** (TVL), which has no clean free tape at all.
- **[Study 561 — ETF-Flow-Momentum](../../561-etf-flow-momentum/)** — the TradFi cousin: do fund
  *flows* predict returns. Study 583 is the DeFi on-chain analogue (TVL is the crypto "flow").
- **[Study 507 — Cross-Sectional-Momentum](../../507-cross-sectional-momentum/)** — the equities
  cross-sectional momentum sort whose ranking machinery this study reuses on a TVL signal.

## Shared method

- **One-sample *t*** on the time series of monthly long-short returns — the headline test.
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null: shuffle the
  TVL-growth labels within each month against forward returns and read the spread's tail
  probability.
- **Cluster-robust (by-month block) OLS** — the pooled protocol-level slope with month-clustered
  standard errors, so within-month cross-sectional dependence does not inflate the *t*.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a robust
  *t* ≥ 2 **on a real tape** for `REAL`; synthetic-only ⇒ at most `WEAK`), one execution lag, and
  costs one-way × NAV with shorts paying borrow.
