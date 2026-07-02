# References & literature map — Study 549 (Spotify-Mood)

## The claim, at full strength

- **Edmans, Fernandez-Perez, Garel & Indriawan (2022)**, *"Music Sentiment and Stock Returns
  Around the World."* *Journal of Financial Economics* 145(2). The flagship result: a country-level
  **music-sentiment index** built from the average musical **valence** (Spotify's happy↔sad audio
  feature) of the most-streamed songs is *positively* related to contemporaneous stock returns and
  *negatively* to future returns (a sentiment-reversal pattern), in a panel of 40 countries. Uses a
  **licensed proprietary** weekly stream + valence dataset — the source a retail stack cannot reach,
  which is exactly why this study's mood tape is synthetic.
- **Baker & Wurgler (2006, 2007)**, *"Investor Sentiment and the Cross-Section of Stock Returns"* /
  *"Investor Sentiment in the Stock Market."* The canonical sentiment-and-returns framework the
  music-sentiment work sits inside: sentiment predicts returns, especially with reversal.
- **Edmans, García & Norli (2007)**, *"Sports Sentiment and Stock Returns."* *Journal of Finance*
  62(4). The sibling mood-proxy result (national-team losses → next-day underperformance) — the same
  "aggregate mood moves the tape" family; see the desk's [Study 300 — Sports-Sentiment](../../300-sports-sentiment/).
- **Hirshleifer & Shumway (2003)**, *"Good Day Sunshine: Stock Returns and the Weather."* *Journal
  of Finance* 58(3). The earliest clean mood-proxy-and-returns paper; the template every alt-data
  mood study (weather, sports, music) follows.

## The data-availability wall (why this study is synthetic-only)

- **Spotify Web API — audio features / audio analysis endpoints.** These are the only public source
  of a track's ``valence``. Spotify **deprecated audio-features access for new applications in
  November 2024** (existing apps grandfathered, no new grants), and the endpoint never returned a
  survivorship-free *historical monthly panel of global top-chart valence*. There is therefore no
  free, reconstructible real valence tape — the mood series here is a seeded synthetic proxy, and
  the SIGNAL axis is capped accordingly (a synthetic input can never clear the REAL |*t*| ≥ 2 bar).

## Neighbours on this bench (the dedup map)

- **[Study 256 — Twitter-Mood](../../256-twitter-mood/)** — Bollen's "Twitter mood predicts the
  market." A *social-media* mood proxy on a curated reconstruction; Study 549 is the *music-valence*
  proxy and is synthetic-only by data availability, not a curated reconstruction.
- **[Study 300 — Sports-Sentiment](../../300-sports-sentiment/)** — the Edmans-García-Norli sports
  mood effect. Same mood-proxy family, different proxy (match results, not stream valence).
- **[Study 257 — AAII-Sentiment](../../257-aaii-sentiment/)** /
  **[Study 335 — Buzz-Sentiment-ETF](../../335-buzz-sentiment-etf/)** /
  **[Study 392 — Glassdoor-Sentiment](../../392-glassdoor-sentiment/)** — other sentiment/alt-data
  proxies on the desk. Study 549 is distinct in its proxy (musical valence) and in being *forced*
  synthetic by the closed Spotify API.

## Shared method

- **Newey & West (1987)** — the heteroskedasticity-and-autocorrelation-consistent (HAC) standard
  error used on the predictive-regression slope, because a persistent mood series induces
  autocorrelation that would inflate a naive *t*.
- **Circular-shift / permutation testing** (Fisher 1935; Politis & Romano 1994 for the circular
  block idea) — the placebo null: shift the valence series against returns and read the slope's tail
  probability.
- **Bonferroni (1936)** — the multiple-comparisons bar applied to the lag-1..5 sweep (the
  Granger-lag data-mining trap).
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (HAC *t* ≥ 2 on
  a **real** tape plus a placebo null and seed-robustness), the synthetic-only cap, one execution
  lag, and costs one-way × NAV with shorts paying borrow.
