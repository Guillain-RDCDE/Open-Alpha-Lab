# Study 134 — Bitcoin-Dominance

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Regression slope is consistently negative (falling dominance -> higher alt spread) but best HAC *t* = −1.88, short of the |*t*| >= 2 bar; no variant clears on conditional mean (*t* = 1.27). |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Even the weak signal yields +0.45%/week gross — but altcoin execution costs (wide spreads, deep slippage), leverage requirements, and extreme drawdown risk in bear markets make net returns negative. |
| **Alt-season timing?** | ![Coincident](https://img.shields.io/badge/Alt--season_timing%3F-Coincident-8b949e?style=flat-square) | Dominance falls *as* alts rise (tautological by construction); the lagged forecast is weak and likely reflects one-cycle regime pattern, not a repeatable edge. |

> **In one sentence:** "alt season" is a real rotation pattern but BTC dominance is a coincident descriptor rather than a tradable forecaster — the lagged signal barely misses the inference bar, the sample spans only one crypto cycle (~6 years), and execution costs on altcoins consume any residual gross edge.

## What we tested

The "alt season" narrative: when Bitcoin's share of total crypto market cap (BTC dominance) falls, capital rotates from BTC into altcoins, and a long-alt/short-BTC trade profits. We operationalise this as: does the *N-day change in BTC dominance* forecast the *forward 1-week alt-minus-BTC spread return*? Dominance is proxied from price × fixed supply scalars (BTC, ETH, XRP, ADA, SOL, BNB, DOGE) using Yahoo Finance daily data from 2020-04-10 (SOL's listing date, which sets the effective panel start) through 2026-06-14. Three signal variants are tested (10-day dominance change, 20-day change, 52-week percentile rank) and pinned against a permutation-shuffle null. A deterministic synthetic panel with a tunable dominance→spread effect serves as the positive control.

The study honestly names its limits: the panel covers only one crypto cycle; the alt basket is survivorship-biased (we use 2024's winners, not a live historical universe); and the dominance proxy uses fixed rather than live circulating supply. Positive results are best interpreted as upper bounds on what any live implementation would find.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the alt-season narrative, the dominance proxy, the fair forecast test, why "alt season" is mostly a coincident label rather than a tradable signal |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | regression slope + HAC SE, horizon sweep, permutation null, the survivorship and proxy biases quantified, the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`bitcoin_dominance/`](bitcoin_dominance/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
