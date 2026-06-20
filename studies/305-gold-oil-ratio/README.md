# Study 305 — Gold-Oil-Ratio

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The *timing* claim (beat buy-and-hold) is rejected in **all 12** robustness cells (*t* vs B&H negative everywhere). The only real thing — beating random timing — is fragile: *t* = +2.30 at one seed but **+1.87 averaged over 20**, below the bar, and a pre-2015 (GFC) artefact. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Trails SPY buy-and-hold by **−2.3%/yr** (8.9% vs 11.2%) for a statistically tied excess-Sharpe (0.59 vs 0.56). No cost level reveals an edge — there isn't one over the passive default. |
| **Does it time equities like Dr. Copper?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | It's a *defensive regime gauge* (halves crash-era drawdown, −29% vs −55%) but not an equity *timer* — it loses to simply holding SPY in every parameterisation. |

> **In one sentence:** the gold/oil ratio reads risk-off in a crisis and will dutifully cut your 2008 drawdown in half — but used as a market-timing switch it underperforms buy-and-hold by 2.3%/yr in every configuration, its edge over a coin-flip averages below *t* = 2, and what little signal there is dies after 2015; Dr. Oil flunks the timing exam.

## What we tested

The folk pitch — a sibling of the copper/gold "Dr. Copper" story — is that the **gold/oil ratio** reads the business cycle: when gold is expensive relative to oil the market is risk-off/contractionary, so you should go defensive, and when oil firms up growth is on, so you should be long equities. We take the strongest *tradable* version: a binary risk-on/risk-off switch that holds SPY when the standardised 60-day gold/oil-ratio deviation is normal and rotates to cash (the T-bill rate) when it spikes high. We race it — on an **excess-of-cash** basis, with **one** execution lag and one-way costs per switch — against buy-and-hold SPY and against a **random-timing control** holding cash on the same fraction of (random) days, over 20 years of GLD/USO/SPY/^IRX daily data (2006–2026). This is deliberately *not* Study 85's predictive regression nor Study 113's metals pairs-trade — it's the "can you trade the regime call?" question.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the Dr.-Copper-for-oil story, why halving your drawdown isn't free, why losing to buy-and-hold matters |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | excess-vs-excess Sharpe race, HAC *t* vs B&H and vs a random-timing control, seed-fragility, the robustness grid, the pre/post-2015 split, the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`gold_oil_ratio/`](gold_oil_ratio/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
