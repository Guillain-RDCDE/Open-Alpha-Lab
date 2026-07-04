# References — Study 632 (Crypto Cross-Sectional Momentum)

## The claim's source

- **Liu, Y., Tsyvinski, A. & Wu, X. (2022).** *Common Risk Factors in Cryptocurrency.*
  **Journal of Finance**, 77(2), 1133–1177. <https://doi.org/10.1111/jofi.13119>
  (working paper: NBER WP 25882, 2019, <https://www.nber.org/papers/w25882>).
  The canonical crypto factor zoo: of size, momentum and a long list of candidate
  factors, **cross-sectional momentum at 1–4 week formation horizons** is one of the
  very few that prices the coin cross-section. Our study is a direct weekly-quintile
  replication of that claim on a transparent 44-coin panel.
- **Liu, Y. & Tsyvinski, A. (2021).** *Risks and Returns of Cryptocurrency.*
  **Review of Financial Studies**, 34(6), 2689–2727.
  <https://doi.org/10.1093/rfs/hhaa113> (NBER WP 24877). Time-series evidence that
  past 1-week returns predict future coin returns — "last week's winners keep winning".

## Key related papers

- **Jegadeesh, N. & Titman, S. (1993).** *Returns to Buying Winners and Selling Losers.*
  Journal of Finance, 48(1), 65–91. The equity momentum template the crypto factor
  imitates (at ~50× the clock speed: weeks, not months).
- **Rohrbach, J., Suremann, S. & Osterrieder, J. (2017).** *Momentum and Trend Following
  Trading Strategies for Currencies and Bitcoin.* SSRN 3081573. Early crypto momentum
  evidence.
- **Newey, W. & West, K. (1987).** *A Simple, Positive Semi-Definite, Heteroskedasticity
  and Autocorrelation Consistent Covariance Matrix.* Econometrica, 55(3), 703–708.
  The HAC *t* used throughout (Bartlett kernel, 4 weekly lags).

## Data sources

- **Binance spot klines** — `GET https://api.binance.com/api/v3/klines` (public, keyless),
  `interval=1w`, weeks open Monday 00:00 UTC. Primary tape; crucially it still serves
  full history for **delisted** pairs (EOS, XMR, WAVES, OMG, NANO, MATIC, FTM, FTT, ANT,
  SRM, LUNA), our survivorship softeners. `LUNAUSDT` concatenates old LUNA and the
  Terra 2.0 relisting under one symbol — we truncate at 2022-05-15, keeping the crash.
- **yfinance** (`pip install yfinance`, Yahoo! Finance `-USD` pairs) — daily closes
  resampled to the same Monday-open weeks, used **only** to backfill a coin's
  pre-Binance-listing history (e.g. DOGE 2017-11 → 2019-07). <https://finance.yahoo.com>

## Sibling desk studies (the dedup map — what this study is *not*)

- [251-crypto-reversal](../../251-crypto-reversal/) — short-horizon cross-sectional
  **reversal** (the opposite sign at the daily/weekly-reversal horizon). This study is
  the **continuation** panel at the Liu-Tsyvinski-Wu 1–4-week horizons.
- [222-altseason-rotation](../../222-altseason-rotation/) — the **BTC-dominance
  rotation** folklore (a two-asset regime trade), not a coin-level cross-section.
- [210-crypto-trend](../../210-crypto-trend/) — **time-series** trend on single assets;
  here the signal is purely **relative** (winners vs losers within the same week).

## Shared method

- Desk house style & inference bar: [`METHODOLOGY.md`](../../../METHODOLOGY.md) —
  HAC *t* ≥ 2 on the real tape for `REAL`; synthetic controls are machinery proofs,
  never evidence; costs one-way × traded NAV with shorts paying borrow; survivorship
  named on the Signal axis.
