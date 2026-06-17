# Study 291 — Doge-Tweets

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Day-0 abnormal return (DOGE net of a market model on BTC) = **+27.9%**; permutation p ≈ **0**; **18/23** events positive; outlier-robust t = **+3.9** (drop the lone +346% "Dogefather" day). Musk's tweets genuinely coincided with large same-day Dogecoin jumps BTC can't explain. *(Hindsight-curated tweet list — an upper bound, named here.)* |
| **Tradability** — does it survive costs, capacity, lag? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The move lands **on the tweet day**, before any daily trader can act. With an honest one-day lag (enter at tweet-day close) the post-tweet drift is +4%/trade at best, net **t ≤ 1**, hit-rate **< 40%** — indistinguishable from zero after 60 bps round-trip. |
| **Busted?** | ![Mostly](https://img.shields.io/badge/Mostly-dab617?style=flat-square) | Real effect, but regime-bound (2021 mania + 2024–25 D.O.G.E.) and gone by the close — you can't buy the spike, only chase it. |

> **In one sentence:** Elon's tweets really *did* move Dogecoin — a robust, distribution-free +28% same-day abnormal jump — but the move was faster than you: lag the entry by one honest day and the edge evaporates.

## What we tested

We hardcode 23 of the most widely reported Musk-Dogecoin events (2019–2025) in
`data.py` and pull DOGE-USD / BTC-USD daily closes from Yahoo (cache-only). We fit a
market model `DOGE = α + β·BTC` on non-event days (so BTC absorbs the broad-crypto
move), measure DOGE's **abnormal** return on and around each tweet day (AAR on day 0,
CAR over [−1, +3]), and test it with a **10,000-draw permutation test** plus an
outlier-robust cross-sectional t. Then we ask the only question that pays: can you
*trade* it? We buy DOGE at the **close of the tweet day** (a one-day execution lag),
hold 1/3/5 days, charge 30 bps one-way × NAV, and benchmark against BTC. A synthetic
positive control (planted +1,500 bps bump) confirms the engine detects a real effect;
the null tape confirms it cries no wolf.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the same-day spike, why BTC is the fair benchmark, why you can't catch it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | market-model abnormal returns, AAR/CAR window, permutation test, outlier robustness, the lagged backtest, the positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`doge_tweets/`](doge_tweets/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
