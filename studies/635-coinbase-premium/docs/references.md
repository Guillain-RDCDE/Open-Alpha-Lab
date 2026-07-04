# References — Study 635 (Coinbase Premium)

## The claim's source

- **CryptoQuant — "Coinbase Premium Gap / Coinbase Premium Index"** —
  <https://cryptoquant.com/asset/btc/chart/market-indicator/coinbase-premium-gap> —
  the on-chain-analytics indicator that popularised the exact construction we test
  (Coinbase BTC-USD minus Binance BTC-USDT), promoted through the 2020-21 bull (Ki
  Young Ju and others) as the tell of the **US institutional bid**: "when Coinbase
  trades rich, institutions are buying — and the market follows."
- Contemporary coverage of the 2020-21 "institutional bid" narrative (Saylor /
  MicroStrategy, Tesla): e.g. CoinDesk market reports citing the Coinbase premium as
  the institutional footprint, 2020-12 → 2021-04.

## Key papers

- **Makarov, I. & Schoar, A. (2020)** — *Trading and Arbitrage in Cryptocurrency
  Markets*, Journal of Financial Economics 135(2) — the canonical study of
  cross-exchange crypto price deviations: large, persistent country/venue premia
  (including the Kimchi premium), closed over time by arbitrage capital; deviations
  co-move with buying pressure. <https://doi.org/10.1016/j.jfineco.2019.07.001>
- **Choi, K.J., Lehar, A. & Stauffer, R. (2022)** — *Bitcoin Microstructure and the
  Kimchi Premium* — the mechanics of a venue premium as a capital-flow bottleneck.
  <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3189051>
- **Griffin, J.M. & Shams, A. (2020)** — *Is Bitcoin Really Un-Tethered?*, Journal of
  Finance 75(4) — why the USDT leg is not a clean dollar: Tether issuance/peg stress
  moves BTCUSDT quotes independently of any USD-venue flow (our named caveat and the
  winsorized robustness leg). <https://doi.org/10.1111/jofi.12903>
- **Lyons, R.K. & Viswanath-Natraj, G. (2023)** — *What Keeps Stablecoins Stable?*,
  Journal of International Money and Finance — USDT/USD peg dynamics; the depeg
  episodes that dominate the 2017-18 tails of our premium series.
  <https://doi.org/10.1016/j.jimonfin.2023.102777>
- **Newey, W.K. & West, K.D. (1987)** — *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica
  55(3) — the HAC *t* used on every slope and mean here.
- **Welch, B.L. (1947)** — *The generalization of "Student's" problem when several
  different population variances are involved*, Biometrika 34 — the group-split test.

## Data sources

- **Coinbase Exchange API** — `GET https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=86400`
  — UTC daily candles for the USD leg (public, keyless; 300 bars/request, paginated).
- **Binance spot API** — `GET https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d`
  — UTC daily klines for the USDT leg (public, keyless; 1,000 bars/request, paginated
  from the 2017-08-17 listing).
- Both cached once to `_cache/btc_cb_bn_daily.parquet`; every run is cache-first.

## Sibling studies on this desk (dedup map)

- [294-coinbase-rank](../../294-coinbase-rank/) — the Coinbase **App-Store rank**
  top-signal. Same venue, entirely different object: that study tests a *retail
  attention* proxy (app downloads); this one tests the **exchange price premium**
  itself — the institutional-flow footprint. New axis, not a re-run.
- [618-gbtc-premium-cycle](../../618-gbtc-premium-cycle/) — the GBTC NAV premium:
  a *fund wrapper* premium, not a spot cross-venue gap.
- [325-crypto-fear-greed](../../325-crypto-fear-greed/) — sentiment index timing on
  the same underlying; different signal family.
- [134-bitcoin-dominance](../../134-bitcoin-dominance/) /
  [632-crypto-xs-momentum](../../632-crypto-xs-momentum/) — BTC-complex context.

## Method citations

- Desk house style: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar
  (HAC *t* ≥ 2 on the real tape for REAL), one-lag execution, one-way costs × traded
  NAV, ≥ 20-seed random baselines, synthetic controls as machinery proofs only.
