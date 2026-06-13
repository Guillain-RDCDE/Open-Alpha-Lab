# Study 91 — Death-Cross ✝️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The timer's Sharpe edge over buy-and-hold is **+0.097**, but the daily return *difference* doesn't clear the bar (HAC *t* = **−0.99**). It does beat an exposure-matched random coin **98%** of the time — so the crossover dates aren't pure noise. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | It **trails buy-and-hold by ~1.2 pts/yr** (9.64% vs 10.82% CAGR, net of 5 bps/switch). The Sharpe gain is just lower equity exposure — *beta you can buy more cheaply* by holding less stock — not skill. Taxes on 31 switches widen the gap. |
| **Dodges crashes?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | Real risk reduction: max drawdown **−33.7%** vs **−55.2%**, volatility **13.7%** vs **18.6%**, and it beats the matched random coin on Sharpe — the dates carry information. |

> **In one sentence:** the death cross genuinely **dodges crashes** — half the drawdown, lower vol, and it beats a matched random coin 98% of the time — but it **does not beat buy-and-hold**, earning ~1.2 points/year *less*, because what it sells as a forecasting signal is really just *less beta*.

## What we tested

The most-quoted rule on financial TV, stated at full strength: *"when the 50-day moving average crosses below the 200-day — the **death cross** — sell; re-enter on the **golden cross**; you'll sidestep the bear markets and beat buy-and-hold."* We take it literally — long SPY (total return) while SMA(50) ≥ SMA(200), in cash (earning 0%, a conservative choice) otherwise — act **one day after** each cross, charge **5 bps** per switch, and pin it against two yardsticks: **buy-and-hold**, and a **matched random-timing control** that holds the same total time in cash, in runs of the same lengths, but on random dates. A deterministic synthetic tape with long planted bull/bear regimes serves as the positive control (the filter banks the edge there; on a driftful random walk it correctly does not).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the rule, the equity curves, the drawdown it really cuts, the matched-coin test, why it still loses to just holding |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC *t* on the return difference, the exposure-matched placebo (200 seeds), the alpha-vs-beta read, capacity |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`death_cross/`](death_cross/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
