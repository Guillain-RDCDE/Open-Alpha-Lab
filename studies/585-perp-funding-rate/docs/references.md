# References & literature map — Study 585 (Perp-Funding-Rate)

## The claim, at full strength

- **Perpetual-swap mechanics (BitMEX 2016; Binance/Bybit funding docs).** A crypto *perpetual
  swap* has no expiry; it is tethered to the spot index by a periodic **funding rate** exchanged
  directly between longs and shorts (typically every 8 hours). When the perp trades *above* the
  spot index the funding rate is **positive** and longs pay shorts; when below, shorts pay longs.
  Funding is therefore a near-direct read on aggregate **positioning and leverage**.
- **The contrarian / positioning folklore.** The widely-repeated desk claim: extreme *positive*
  funding marks an over-crowded, over-levered long book that is vulnerable to a long-squeeze /
  liquidation cascade — a **contrarian short** signal for the forward return — and extreme
  *negative* funding marks capitulated shorts, a contrarian long. It is a mean-reversion /
  positioning signal, not a trend signal. Popularised by exchange and data-vendor research desks
  (Coinglass funding heatmaps; Amberdata, Kaiko, Glassnode research notes; countless crypto-desk
  and trader write-ups). No single canonical academic paper owns the exact "extreme funding =
  contrarian" rule — it is folklore built on the mechanics above.
- **Related academic evidence.** Work on crypto perpetuals shows funding co-moves with sentiment,
  leverage and liquidation risk, and that the perpetual **basis** carries information about
  positioning (e.g. studies of the crypto futures basis and funding-rate arbitrage). The
  *contrarian return-predictability* of funding specifically is closer to practitioner lore than to
  a settled academic result — which is exactly why the desk treats it as a claim to test, not a
  fact.

## Why the core is synthetic

- The reproducible test would need the **historical funding-rate tape** paired with BTC/ETH forward
  returns. Funding history is **paid or rate-limited** exchange/derivatives data (Binance/Bybit
  `fapi` funding endpoints, Coinglass, Amberdata, Kaiko, Glassnode). A no-key retail stack
  (yfinance) reaches BTC/ETH *prices* but **not** funding. So the engine is validated on a
  deterministic synthetic funding + forward-return panel with one planted-effect knob
  (`contrarian_beta`), and the data-availability limitation is stated openly on the SIGNAL axis. A
  synthetic-only study can never earn `REAL` (that needs a robust *t* ≥ 2 on a real tape) — the same
  house rule applied to the desk's [273 Lego-Returns](../../273-lego-returns/),
  [275 Whisky-Cask](../../275-whisky-cask/) and [276 Sneaker-Resale](../../276-sneaker-resale/)
  synthetic-data studies.

## Neighbours on this bench (the dedup map)

- **[Study 133 — Crypto-Seasonality](../../133-crypto-seasonality/)**,
  **[175 Crypto-Weekend](../../175-crypto-weekend/)**,
  **[210 Crypto-Trend](../../210-crypto-trend/)**,
  **[251 Crypto-Reversal](../../251-crypto-reversal/)**,
  **[325 Crypto-Fear-Greed](../../325-crypto-fear-greed/)** — the desk's crypto studies. Those test
  **price/calendar/sentiment** signals on the *free price tape*; Study 585 is the one crypto signal
  whose entire input (**funding rate**) is *not* free — hence the synthetic core and the explicit
  data caveat. Distinct signal, distinct data-availability story.
- **[Study 273 — Lego-Returns](../../273-lego-returns/)** / **[275 Whisky-Cask](../../275-whisky-cask/)**
  / **[276 Sneaker-Resale](../../276-sneaker-resale/)** — the desk's other **synthetic-data-only**
  studies (the real free tape does not exist), the tone this study matches on the SIGNAL axis.

## Shared method

- **Welch (1947)** — the unequal-variance two-sample *t* used for the cold-minus-hot bucket spread.
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null: shuffle the
  funding labels against forward returns and read the spread's tail probability.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (`REAL` needs a
  robust *t* ≥ 2 on a **real** tape; synthetic-only caps at `WEAK`/`NONE`), the seed-robust
  synthetic control (≥ 20 seeds), one execution lag, and costs one-way × NAV with the short leg
  paying its carry.
