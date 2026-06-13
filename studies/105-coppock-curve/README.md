# Study 105 — Coppock-Curve

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | 12m gross mean **+16.1% per signal** (vs +7.8% BaH), HAC *t* = **+5.74** — but n = **19 signals in 76 years**; bear-market beta explains much of the gap. |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | ~1 signal per 4 years; single-signal risk (Dec 2001: −26.6% in 12m); lag of 2–12 months after the actual trough; no sell rule. |
| **Beats buy-and-hold?** | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | 18/19 signals positive at 12m; Coppock +16.1% vs BaH +7.8% (two-sample *t* = +2.57, *p* = 0.019) — but entry at depressed prices is the confound. |

> **In one sentence:** the Coppock Curve identifies genuine bear-market trough entries — but the signal fires only 19 times in 76 years, arrives 2–12 months late, and its apparent edge is partly explained by buying cheap after crashes rather than forecasting recoveries.

## What we tested

The Coppock Curve (Edwin Coppock, Barron's 1962): a 10-month weighted moving average of (ROC(14) + ROC(11)) on monthly S&P 500 closes, with a buy signal when the curve turns up from below zero. We steelman the claim as: *"Coppock buy signals deliver significantly higher 6- and 12-month forward returns than random-timing entries drawn at the same frequency, and exceed the buy-and-hold average."* We test on **^GSPC monthly since 1950** (918 months, 19 signals) and SPY since 1993, comparing forward log returns vs a **random-timing control** (same n, randomly selected months) and **buy-and-hold** (all months). A deterministic synthetic tape with a tunable market cycle serves as the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the grief-counselling origin story, the 19-signal chart, why the Dec 2001 failure matters, the bear-market-cheap confound in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stats, drawdown decomposition, 30-seed random control stability, synthetic cycle-strength sweep, lag table at historical troughs |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`coppock_curve/`](coppock_curve/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
