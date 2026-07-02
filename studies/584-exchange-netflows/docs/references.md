# References & literature map — Study 584 (Exchange-Netflows)

## The claim, at full strength

- **CryptoQuant — "Exchange Netflow" / "Exchange Reserve" indicators.** The canonical desk
  statement of the folklore: rising exchange **net-inflow** (deposits − withdrawals) signals coins
  being staged for sale (bearish); net-**outflow** signals accumulation to cold storage (bullish).
  Sold as a paid on-chain analytics product (address-clustering into exchange wallets).
- **Glassnode — on-chain flow metrics** (Exchange Net Position Change, Exchange Inflow/Outflow
  Volume). The same net-flow concept with a defended exchange-address taxonomy; the reference
  provider for "coins on exchanges" analytics, again behind a paid tier.
- **Nansen — "Smart Money" / exchange-flow labels.** Entity-labelled on-chain flow (wallets tagged
  to exchanges and funds); the labelling *is* the moat and the reason a free stack cannot rebuild
  a netflow series.

## The academic backdrop (does on-chain data predict crypto returns?)

- **Liu & Tsyvinski (2021)**, *"Risks and Returns of Cryptocurrency."* *Review of Financial
  Studies* 34(6). Establishes that crypto returns are driven largely by momentum and
  investor-attention proxies; sets the bar for what an on-chain predictor must beat.
- **Liu, Tsyvinski & Wu (2022)**, *"Common Risk Factors in Cryptocurrency."* *Journal of Finance*
  77(2). A three-factor (market, size, momentum) model for crypto — the factor benchmark any
  on-chain "signal" is measured against.
- **Makarov & Schoar (2020)**, *"Trading and Arbitrage in Cryptocurrency Markets."* *Journal of
  Financial Economics* 135(2). Documents cross-exchange flows and arbitrage; shows exchange-level
  on-chain movement is real and measurable — for those with the address labels.
- **Griffin & Shams (2020)**, *"Is Bitcoin Really Un-Tethered?"* *Journal of Finance* 75(4). A
  landmark on-chain forensic study — and a caution that on-chain "flow" narratives are easy to
  assert and hard to identify cleanly.

## The data-availability wall (why this study is synthetic-only)

- A BTC **exchange net-inflow** series is *defined* by which on-chain addresses belong to which
  exchange. That mapping is built by proprietary clustering (Glassnode, CryptoQuant, Nansen,
  Chainalysis) and sold behind paid API keys with no usable free history. The public blockchain
  exposes transfers, **not** the exchange labels — so a no-key retail stack cannot reconstruct an
  honest netflow tape. Per house rule, a study with no real tape is capped at **WEAK/NONE**; this
  one is built on a deterministic synthetic world and stamps `NONE` (see
  [`../../METHODOLOGY.md`](../../../METHODOLOGY.md) → *The inference bar*: `REAL` is earned by the
  tape, not by the literature or a synthetic control).

## Neighbours on this bench (the dedup map)

- **[Study 133 — Crypto-Seasonality](../../133-crypto-seasonality/)**,
  **[Study 175 — Crypto-Weekend](../../175-crypto-weekend/)**,
  **[Study 210 — Crypto-Trend](../../210-crypto-trend/)**,
  **[Study 251 — Crypto-Reversal](../../251-crypto-reversal/)**,
  **[Study 325 — Crypto-Fear-Greed](../../325-crypto-fear-greed/)** — the crypto studies that run
  on the *price/sentiment* tape a free stack *can* reach. Study 584 is the one crypto claim whose
  underlying data (exchange-labelled on-chain flow) is **paywalled**, so it is synthetic-only — the
  distinction it carries on the SIGNAL axis.

## Shared method

- **Welch (1947)** — the unequal-variance two-sample *t* for the high-inflow vs low-inflow forward
  return spread.
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null: shuffle the
  net-inflow labels against forward returns and read the spread's tail probability.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a robust *t* ≥ 2
  on a *real* tape for `REAL`; synthetic control is a machinery proof, never market evidence), one
  documented execution lag, gross AND net labelled, shorts paying borrow, and the data-availability
  caveat named on the SIGNAL axis.
