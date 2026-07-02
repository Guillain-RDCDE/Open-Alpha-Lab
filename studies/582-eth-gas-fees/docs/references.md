# References & literature map — Study 582 (ETH-Gas-Fees)

## The claim, and where it comes from

- **On-chain "euphoria" folk claim** (crypto Twitter, Glassnode / Dune / Nansen dashboards, ~2020-).
  Gas fees are the market-clearing price of Ethereum blockspace; a spike means the chain is
  congested with mints, DEX swaps, leverage and memecoin manias. The contrarian reading: that
  pay-anything congestion marks a local *top*, so forward returns should be low after a gas spike.
  The bullish reading: high gas is *genuine demand* for blockspace and is neutral-to-positive.
  There is no single canonical academic statement — it is a widely repeated analytics-blog heuristic.
- **Buterin et al. — EIP-1559** (2019, activated in the *London* hard fork, Aug 2021),
  *"Fee market change for ETH 1.0 chain."* Replaced the first-price gas auction with a burned
  **base fee** plus a **priority tip**, so the meaning and dynamics of "gas price" change sharply
  in Aug 2021 — a regime break any real study must handle. Named on the data-availability caveat.
- **Ethereum rollup / L2 migration (2023-2025).** Arbitrum, Optimism, Base and friends moved most
  transaction volume off mainnet, so post-2023 mainnet gas measures a *different* activity base than
  2021 — a second regime break, and a further reason a clean real gas tape is hard.

## Why this study is synthetic-only (the data-availability limit)

- A daily, survivorship-free Ethereum gas-fee series back through the 2021 mania requires an
  **archive node** or a **keyed API** (Etherscan, Dune Analytics, Owlracle, Blocknative). A no-key
  retail stack (yfinance) reaches **ETH-USD prices** but not gas. So — exactly as the desk treats
  its other data-starved claims — this study builds a deterministic synthetic gas/price world and
  states the limit on the SIGNAL axis. A synthetic-only study can never earn a `REAL` stamp (that
  needs a robust *t* ≥ 2 on a REAL tape). Cousins on this bench that do the same:
  **[273 Lego-Returns](../../273-lego-returns/)**, **[275 Whisky-Cask](../../275-whisky-cask/)**,
  **[276 Sneaker-Resale](../../276-sneaker-resale/)**.

## Neighbours on this bench (the dedup map)

- **[325 Crypto-Fear-Greed](../../325-crypto-fear-greed/)** — the closest cousin: a *sentiment* gauge
  read contrarian for forward BTC returns. Study 582 uses an **on-chain activity** proxy (gas) rather
  than a survey/sentiment index, and is synthetic-only because the gas tape is unreachable.
- **[292 Bitcoin-Hashrate](../../292-bitcoin-hashrate/)** / **[295 Stablecoin-Supply](../../295-stablecoin-supply/)**
  — other on-chain metrics read as forward-return predictors. Same family (on-chain → price), a
  different metric (miner economics / stablecoin float, not blockspace congestion).
- **[133 Crypto-Seasonality](../../133-crypto-seasonality/)** / **[251 Crypto-Reversal](../../251-crypto-reversal/)**
  — price-only crypto-timing studies. Study 582's signal is an *on-chain fundamental*, not a price
  pattern.

## Shared method

- **Welch (1947)** — the unequal-variance two-sample *t* for the spike-day vs calm-day forward-return
  gap.
- **Newey & West (1987)** — the HAC standard errors on the forward-return slope and the overlay mean,
  robust to the autocorrelation and volatility clustering of daily crypto returns.
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null: permute the
  gas-spike labels against forward returns and read the gap's tail probability.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a robust
  *t* ≥ 2 on a REAL tape for `REAL`; synthetic-only caps at `NONE`/`WEAK`), one documented execution
  lag, gross-and-net labelling, shorts paying borrow, and the seed-robust (≥ 20 seeds) synthetic
  positive control.
