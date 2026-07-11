# Study 688 — Concealing Baby Swallow 🕯️🐦

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the pattern mark a bullish reversal? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Across **111 tickers** and **4,957 stock-years** of daily bars (up to 64.5 years each), the plain reading of the four-candle shape fires **4** times, ever; the literature-close cut fires **0** times. Both sit below the pre-registered floor of **8** pooled events — no *t*-statistic is computed. Not "tested and failed" — **too rare to test at all**. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Nothing to charge costs against. Even granting every benefit of the doubt to the 4 loose-cut hits, that's roughly one signal every 15 years pooled across the desk's largest single-pattern basket — no venue, no capacity, no book. |
| **"Too rare to ever test on real markets?"** | ![Confirmed](https://img.shields.io/badge/Too_rare_to_test%3F-Confirmed-8b949e?style=flat-square) | A synthetic control proves the detector is unbiased — it stays quiet on 19/20 null seeds (nominal 5% rate) and recovers a hand-planted reversal cleanly (*t* = 14.4). The real tape's near-zero count is a property of the market, not a broken or over-strict rule. |

> **In one sentence:** we scanned 111 US stocks and ETFs across up to 64.5 years of daily bars for the rarest bullish reversal candle in the Japanese candlestick canon — the four-candle "concealing baby swallow" — and found it **4 times** on a loose reading and **0 times** on the literature-accurate one, well below the sample size needed to say anything statistically; a proven-honest synthetic control confirms the detector works, so the honest verdict is that this claim is **too rare to ever falsify or confirm** on live markets.

## What we tested

The **concealing baby swallow**: two black marubozu, a third black candle that gaps down then rallies intraday into the second candle's real body (a long upper shadow — the failed rally the pattern "conceals") before closing at a new low, then a fourth black candle that totally engulfs the third — including its shadow — and closes at a fresh low again. Folklore reads this as capitulation, the last sellers giving up right before a reversal. We built two OHLC detectors (a loose, practical-chartist cut and a strict, literature-close cut) and scanned every bar of a **111-ticker** basket (SPY/QQQ/DIA/IWM + 107 long-listed US large-caps, yfinance daily OHLCV, cache-first, as-of 2026-06-30) — the widest net and longest history the desk has cast for any single named pattern. A pre-registered floor of **8 pooled events** gates whether any *t*-statistic is computed at all; a deterministic synthetic control (exact planted geometry, tunable post-pattern bounce, 20-seed null check) proves the search machinery is sound. **Dedup:** [408-three-black-crows](../408-three-black-crows/) (three candles, bearish, common enough to test — and loses shorted), [186-morning-star](../186-morning-star/) (three candles, common enough to test — and underperforms random days), and [687-ladder-bottom](../687-ladder-bottom/) (five candles, the desk's other very-large-basket rarity case, without this study's precise overlap/engulf geometry) — none of them hit this study's wall: an essentially unusable sample size. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the pattern claims, why we scanned 111 stocks for it, the four occurrences found in full, and the proof the search itself isn't broken |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the loose/strict detector geometry, the pre-registered small-*n* stopping rule, the base-rate-matched event study, the descriptive-only placebo, and the 20-seed synthetic faithful-engine control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`concealing_baby_swallow/`](concealing_baby_swallow/). Basket is **survivors** (all 111 names trade today) — named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
