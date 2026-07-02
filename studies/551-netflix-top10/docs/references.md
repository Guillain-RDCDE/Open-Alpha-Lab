# References & literature map — Study 551 (Netflix-Top10)

## The claim, at full strength

- **Netflix Top-10 / Tudum engagement reports** (netflix.com/tudum, top10.com). Netflix publishes
  a weekly Top-10 with *hours viewed* (later a rolling-window *views* metric after a 2023
  methodology change). The alt-data thesis: rising engagement momentum leads subscriber growth,
  ad inventory and pricing power, so it should *predict* NFLX (and, by spillover, consumer-
  discretionary) returns. The claim this study tests.
- **Da, Engelberg & Gao (2011)**, *"In Search of Attention."* *Journal of Finance* 66(5). The
  canonical "internet-search / attention alt-data predicts returns" result (Google SVI) — the
  template for engagement-as-signal claims like this one, and a reminder that most such edges are
  small, short-lived, or already priced.
- **Chen, De, Hu & Hwang (2014)**, *"Wisdom of Crowds: The Value of Stock Opinions Transmitted
  Through Social Media."* The broader alt-data / crowd-signal literature the streaming-engagement
  idea sits inside.

## The data-availability wall (why synthetic-only)

- The public Top-10 hours series begins only in 2021, changed methodology (*hours viewed* → a
  rolling *views* window) in 2023, and is distributed as PDFs / a JavaScript dashboard rather than
  a free, machine-readable, point-in-time feed. There is **no research-grade real tape** a no-key
  retail stack can pull, so this study is synthetic-only and capped at `WEAK`/`NONE` — the same
  posture as the desk's collectible/alt-data synthetic studies.

## Shared method

- **Newey & West (1987)** — the heteroskedasticity- and autocorrelation-consistent (HAC) standard
  error used for the predictive slope. **Essential here** because weekly signals against multi-week
  *forward* returns produce **overlapping windows**; the naive OLS *t* is inflated, and the NW
  correction is the honest stat (the study's central lesson).
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null: shuffle the
  engagement-momentum labels against forward returns and read the slope's tail probability.
- **Overlapping-return inference** (Hansen & Hodrick 1980; Britten-Jones, Neuberger & Nolte 2011) —
  the reason a single-seed OLS *t* can clear ±2 on pure noise, and why seed-averaging + HAC is
  required.

## Neighbours on this bench (the dedup map)

- **[Study 257 — AAII-Sentiment](../../257-aaii-sentiment/)** / **[Study 335 —
  Buzz-Sentiment-ETF](../../335-buzz-sentiment-etf/)** / **[Study 392 —
  Glassdoor-Sentiment](../../392-glassdoor-sentiment/)** — other *alt-data / sentiment-predicts-
  returns* claims. Study 551 is the **streaming-engagement** variant, and its distinguishing lesson
  is the **overlapping-window false positive**, not sentiment polarity.
- **[Study 273 — Lego-Returns](../../273-lego-returns/)** / **[Study 275 — Whisky-Cask](../../275-whisky-cask/)**
  / **[Study 276 — Sneaker-Resale](../../276-sneaker-resale/)** — the desk's other **synthetic-only,
  no-real-tape** studies (capped at `WEAK`/`NONE`), whose data-availability posture Study 551 shares.

- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a robust
  *t* ≥ 2 on a real tape for `REAL`, plus a placebo null and seed-robustness), the synthetic-only
  cap, one execution lag, and costs one-way × NAV with shorts paying borrow.
