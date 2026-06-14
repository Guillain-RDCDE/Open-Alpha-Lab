# Study 148 — Lunar-Effect

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Contrast (new − full) = +1.06 bps/day, HAC *t* = +0.75 on 24,727 trading days (1928–2026). No sub-period clears the bar; permutation p = 0.24. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Long-new / short-full overlay Sharpe = +0.07 ([−0.12, +0.25]), vs buy-and-hold +0.42 — destroyed by any transaction cost at ~24 round-trips/yr. |
| **Post-publication persistence** | ![None](https://img.shields.io/badge/None-8b949e?style=flat-square) | Out-of-sample (2002+) contrast +1.97 bps, HAC *t* = +0.73 — 25 years of data can't establish the effect. |

> **In one sentence:** the famous "moonstruck investors" effect — lower returns near full moons than new moons — is pure noise on the S&P 500: nearly 100 years of data yield a contrast of just +1 bps/day at HAC *t* = 0.75, the decade sign flips five times out of ten, and the overlay strategy is demolished by buy-and-hold.

## What we tested

Yuan, Zheng & Zhu (2006, *Journal of Empirical Finance*) studied 48 countries over 1973–2001 and found stock returns run roughly 3–5 bps/day higher in the ±7 trading days around a new moon than around a full moon. The proposed channel: full moonlight disrupts sleep, worsens mood, and nudges investors toward risk aversion, briefly suppressing equity demand. We steelman the claim literally: the lunar phase — computed from pure astronomy (the synodic month of 29.53 days anchored to the J2000.0 new-moon epoch, no data fetch) — partitions every trading day into a NEW or FULL window, and we test whether that partition predicts daily returns on the S&P 500 from 1928 to 2026.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the moonstruck story, why the claim sounds plausible, the honest contrast vs buy-and-hold in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stats, decade breakdown, permutation null, long-new/short-full Sharpe, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`lunar_effect/`](lunar_effect/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
