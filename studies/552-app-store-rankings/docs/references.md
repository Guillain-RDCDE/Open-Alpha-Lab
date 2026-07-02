# References & literature map — Study 552 (App-Store-Rankings)

## The claim, at full strength

- **Alt-data folklore / practitioner claim.** A public consumer-tech company's **App Store download
  rank** is a real-time proxy for demand (units, subscriptions, engagement), so when its app climbs
  the charts the stock should climb with it — ranking momentum as a *fundamental nowcast*. This is
  the broad, cross-sectional version of a single-name rank omen.

## Alternative data & the nowcasting mechanism

- **Da, Engelberg & Gao (2011)**, *"In Search of Attention."* *Journal of Finance* 66(5). Google
  search volume (SVI) as a direct, real-time measure of retail attention that predicts short-run
  prices — the canonical "web/behavioural alt-data nowcasts fundamentals/returns" result the app-rank
  claim rides on.
- **Choi & Varian (2012)**, *"Predicting the Present with Google Trends."* *Economic Record* 88.
  The nowcasting programme: high-frequency web signals track contemporaneous economic activity. App
  download rank is the same idea for a specific product.
- **Froot, Kang, Ozik & Sadka (2017)**, *"What Do Measures of Real-Time Corporate Sales Tell Us
  About Earnings Surprises and Post-Announcement Returns?"* *J. Financial Economics* 125. Real-time
  consumer-transaction data forecasts sales and drift — the strongest evidence that a genuine demand
  nowcast (of which app rank is a proxy) can carry return information.
- **Grinblatt & Keloharju (2000)** and the broader **information-coefficient** tradition (Grinold &
  Kahn, *Active Portfolio Management*): the cross-sectional Spearman IC between a signal and forward
  returns, and its *t*-stat over time, is the standard alt-data signal-quality metric — the headline
  test in this study.

## Why app rank is a *noisy* read (the certification hazard)

- App Store rank is **discrete, capped at #1, and mean-reverting near the top**, so the mapping from
  true demand to observed rank is compressed and non-linear.
- An **app is not a ticker**: many chart-toppers are private, and a public parent's revenue is only
  partly the app (ride-hail vs delivery; a family of apps; ads vs subscriptions). The signal-noise
  sweep in [`docs/results.md`](results.md) shows the same true effect falling below the *t* ≥ 2 bar
  as this read gets blurrier — a concrete reason a real mechanism can fail to certify.

## Data availability

- **Apple** publishes only a *live* top-charts snapshot — no official historical rank API.
- The usable rank history is **vendor-gated and modelled** (App Annie / data.ai, Sensor Tower,
  Apptopia): expensive, licensed, and itself an estimate, not a clean tape. So no
  free/survivorship-clean/point-in-time panel exists for a retail stack — this study is
  **synthetic-only**, capped at `WEAK` (a `REAL` stamp requires a robust *t* ≥ 2 on a real tape).

## Neighbours on this bench (the dedup map)

- **[Study 294 — Coinbase-Rank](../../294-coinbase-rank/)** — the single-name, *contrarian-top* app
  rank omen (Coinbase spikes toward #1 → BTC fades). Study 552 is the **broad cross-sectional**
  version: sort *many* consumer-tech names by ranking *improvement* and measure an IC, not one
  event-study on one asset.
- **Sentiment / alt-data cousins** — [257 AAII-Sentiment](../../257-aaii-sentiment/),
  [335 Buzz-Sentiment-ETF](../../335-buzz-sentiment-etf/), [392 Glassdoor-Sentiment](../../392-glassdoor-sentiment/):
  the same "does a soft/behavioural alt-data series predict returns?" question on different feeds.
- **Synthetic-only collectible/alt studies** — [273 Lego-Returns](../../273-lego-returns/),
  [275 Whisky-Cask](../../275-whisky-cask/), [276 Sneaker-Resale](../../276-sneaker-resale/): the
  house pattern of an honest, deterministic synthetic study when no free real tape exists.

## Shared method

- **Spearman rank correlation** (the information coefficient) and the **IC *t*-stat** (mean monthly
  IC / SE) — the alt-data signal-quality bar.
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null: shuffle the
  signal against forward returns within each month and read the mean IC's tail probability.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a robust *t* ≥ 2
  on a **real** tape for `REAL`; literature + a working engine alone is `WEAK`), the seed-robust
  synthetic control (≥ 20 seeds), one execution lag, gross/net labelling, and shorts paying borrow.
