# Study 353 — Smart-Money-Concepts: do "order blocks" and "fair-value gaps" leave footprints? 🕵️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do FVGs fill / OBs bounce better than random? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | SPY's 2,337 fair-value gaps fill **84%** within 20 bars — but a matched random zone fills **86%** and a random walk **84%** (real−placebo *z* = −2.15, the *wrong* sign). Order-block forward returns are **−0.11%** (t = −1.04) and no different from random entry (Welch t = −0.78, p = 0.44). |
| **Tradability** — is there an edge to harvest? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The signal is already absent gross, so costs are academic: the OB book is negative gross and −0.21% net of 5 bp/leg (t = −1.96, wrong sign). Nothing to scale. |
| **Footprints?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | A driftless random walk produces **the same** FVG count, the same 84% fill-rate, and the same null OB "edge" — so an order block or a gap is not evidence of "smart money", it is the geometry of any random walk. |

> **In one sentence:** ICT "Smart Money Concepts" describes real chart shapes — gaps and bounces — that a coin-flip random walk produces just as often, fills just as fast, and reverses no differently, so there is no measurable footprint and nothing to trade.

## What we tested

The huge YouTube/TikTok meme **ICT / Smart Money Concepts (SMC)** claims institutions leave detectable *footprints*: price "respects order blocks" and "fills fair-value gaps". We reduced it to two falsifiable claims and tested them on **33 years of SPY daily OHLC** (8,404 bars, 1993→2026): **(a)** do 3-bar fair-value gaps (a void where candle 1 and candle 3 don't overlap) fill *more / faster* than a random same-size price zone? and **(b)** do forward returns from an "order block" zone *beat* returns from random levels? The decisive control is the null the meme forgets — **a random walk also produces gaps and bounces** — so the Signal axis is the *excess* over a matched random walk and a within-tape placebo, never the raw fill-rate. (Same lesson as [Study 301](../../301-triple-rsi/) and [Study 351](../../351-btc-5m-polymarket-momentum/): an impressive headline number that the right null reproduces for free.)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why an 84% "fill-rate" feels like proof and isn't, the random-walk twin, and why an order block bounces no better than a coin flip — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | FVG fill-rate vs a placebo two-proportion z and a matched random walk, OB forward returns vs random entry (Welch t), a horizon sweep, costs on NAV, and the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`smart_money_concepts/`](smart_money_concepts/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
