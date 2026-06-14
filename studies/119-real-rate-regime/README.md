# Study 119 — Real-Rate-Regime

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Corr(real rate level, fwd 12m return) = **+0.15**; corr(12m chg, fwd return) = **+0.09** — *both positive*, the wrong sign. Q4 (highest real rates) returns **+7.9%/yr** next year; Q1 (lowest) returns **−0.6%/yr**. The relationship is **monotone upward**. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Timing rule earns **−0.2%/yr** vs buy-and-hold at **+2.4%/yr**; underperformance is statistically significant (HAC *t* = **−2.74**) before any costs are charged. |
| **"Don't fight the Fed"?** | ![Backfires](https://img.shields.io/badge/Don't_fight_the_Fed%3F-Backfires-8b949e?style=flat-square) | Rising real-rate regimes return **+6.4%/yr** (*t* = +4.55); falling regimes only **+2.1%/yr** (*t* = +1.58). The rule exits during expansions and stays in during crises. |

> **In one sentence:** 150 years of Shiller data refute "don't fight the Fed" at the real-long-rate level — equities do *better* when real rates are high or rising, the timing rule significantly underperforms buy-and-hold, and the economic story (growth and rates rising together) is consistent across every sub-period.

## What we tested

A staple of market commentary: when the real long-term interest rate (Shiller nominal long rate minus trailing CPI inflation) is high or rising, equities face a discount-rate headwind and investors should step aside. We steelman this with the Gordon Growth Model discount-rate channel and run three tests on 1,797 months of Shiller data (1873–2023): a quartile **level sort**, a **rising/falling regime sort**, and a mechanical **timing overlay** (risk-off when the 12-month real-rate change is positive) benchmarked against buy-and-hold. A deterministic synthetic tape with a tunable regime effect serves as the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the counter-intuitive results in plain language, why the growth-channel confound flips the sign |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | level sort with HAC *t*, momentum sort, timing overlay vs BaH, correlations, sub-period robustness, positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`real_rate_regime/`](real_rate_regime/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
