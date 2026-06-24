# Study 428 — Stochastic RSI

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Net Sharpe 0.31 (HAC *t* +2.03) is harvested equity beta (β = 0.58); **timing alpha −2.66%/yr** (*t* −1.67), and a same-exposure coin beats the rule **84%** of the time. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Loses the Sharpe race to plain buy-and-hold (0.31 vs **0.64**); the gap is negative in **99.9%** of block bootstraps and no cost level closes it. |
| **Adds over RSI?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | ΔSharpe vs plain RSI = **−0.03**, 95% CI [−0.33, +0.24] — a statistical tie. The second oscillator is decoration. |

> **In one sentence:** stacking a Stochastic on top of RSI feels "more sensitive," but on 33 years of SPY the long-flat rule's only positive return is the slice of the equity premium it captures by being long half the time — it times worse than a coin, and adds nothing over plain RSI.

## What we tested

Chande & Kroll's Stochastic RSI (*The New Technical Trader*, 1994) applies the Stochastic
transform to the RSI series, the "indicator of an indicator." The folk rule: %K below 20 =
oversold (buy), above 80 = overbought (exit/short). We turn that into a stateful long-flat
(and long-short) daily timing rule on SPY (1993–2026, total-return bars), enter with one day
of execution lag, charge 1 bp one-way × NAV, and race it **excess-of-cash** against three
benchmarks: buy-and-hold, the *obvious simpler* plain RSI(14), and an SMA(50/200) trend
filter — plus a same-time-in-market random-timing coin to isolate pure timing skill. A
5-ETF panel and a synthetic AR(1) positive control round it out.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "an oscillator on an oscillator" sounds smart, the buy-and-hold race, the coin that times better |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the timing-alpha regression, HAC *t*, block-bootstrap Sharpe gaps, random-timing control, panel, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`stochastic_rsi/`](stochastic_rsi/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
