# Study 431 — Schaff Trend Cycle 📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the timing edge real? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The STC long/flat rule on SPY (33 years) earns a net Sharpe of **0.29** with a HAC **t = 1.87** (under the 2.0 bar), and a block-permutation placebo beats it **96%** of the time. The timing carries no detectable information. |
| **Tradability** — does it survive costs & scale? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Net Sharpe **0.29** vs buy-and-hold's **0.65** (CAGR **2.95%** vs **10.90%**) for no extra safety — it *destroys* value versus doing nothing. Even at **zero** cost it never reaches passive; the long/short version loses money outright. |
| **A "faster MACD"?** | ![Busted](https://img.shields.io/badge/Faster_MACD%3F-Busted-8b949e?style=flat-square) | STC Sharpe **0.29** < the plain MACD crossover it derives from (**0.42**) on SPY, and STC < MACD on **5 of 6** instruments. The whole selling point fails. |

> **In one sentence:** the Schaff Trend Cycle does turn faster than the MACD — but on the S&P 500 that just means more whipsaws, so its long/flat rule lands at net Sharpe 0.29 (HAC *t* = 1.87, placebo *p* = 0.96), losing to buy-and-hold (0.65) *and* to the very MACD it claims to beat (0.42), with the long/short version losing money outright.

## What we tested

The STC is a *double stochastic of the MACD line*, bounded to 0–100 and marketed as a **"faster MACD"** — it snaps to the new trend sooner, so (the pitch goes) you out-earn both the laggy MACD and a passive buy-and-hold. We take that literally: compute the canonical STC (fast 23, slow 50, cycle 10) on **SPY** daily (total-return adjusted, 1993→2026) plus a 6-name robustness panel, go **long** on a cross up through 25 and **flat** on a cross down through 75 (and a long/short variant), enter with **one** execution lag, charge **net** costs one-way × NAV, and race the result — **excess-vs-excess** — against buy-and-hold *and* the plain MACD crossover it claims to improve. The Signal axis is a HAC *t* of the daily net excess return plus a 2,000-draw block-permutation placebo; a planted-trend synthetic control proves the engine *can* bank a real trend edge (so the SPY null is a true negative, not a broken harness).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what STC is, the "faster MACD" pitch in plain language, the equity-curve gap vs buy-and-hold, the MACD head-to-head, and why faster bought whipsaws not profit |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the double-stochastic construction, HAC *t* + block-permutation placebo, excess-vs-excess Sharpe race, cost sweep, the 6-name panel, and a planted-trend faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`schaff_trend_cycle/`](schaff_trend_cycle/). Closes are **total-return adjusted**; all Sharpe races are **excess-vs-excess**, net of 1 bp one-way × NAV with one execution lag. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
