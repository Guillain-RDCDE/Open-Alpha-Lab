# Study 308 — Cocoa-Squeeze 🍫

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | One blow-off, no cross-section. On the real tape "ride it" earns HAC *t* = **+0.08**, "fade it" earns *t* = **−1.99** (a *loss*). The synthetic control proves the engine works — it can never certify the market. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Momentum is a coin flip gross and negative at the first basis point of cost; the fade's block-bootstrap CI is **entirely below zero** once shorts pay borrow and eat the bear rallies. |
| **A repeatable cocoa edge?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | A −77.7% give-back *looks* shortable in hindsight, but it was a once-in-a-generation crop-failure supply shock — not a recurring, harvestable pattern. |

> **In one sentence:** cocoa's 3× parabola into December 2024 and its −78% crash make a spectacular chart and a terrible strategy — riding it earned nothing, fading it lost money, and a single event can't certify a signal no matter how vertical it looks.

## What we tested

After cocoa (`CC=F`) went vertical in 2024 — ~$4,100/t to an all-time high near $12,600/t, then giving back three-quarters of the run — two folk reactions compete: **ride the squeeze** (trend-momentum says a market making new highs keeps going) and **fade the blow-off** (overreaction says grotesque parabolas snap back). We take both literally on cocoa front-month daily bars since 2000, locate the blow-off by a volatility-scaled "stretch above trend," run each leg with one execution lag, HAC *t*-stats, block-bootstrap CIs, costs one-way × turnover × NAV and shorts paying borrow. A deterministic synthetic blow-off — a planted parabola that mean-reverts vs a pure random walk — is the positive control that proves the engine *can* detect a tradable squeeze when one exists.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the chart that fools everyone, why "ride it" and "fade it" both fail, and why one event proves nothing |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC *t*, block-bootstrap CIs, cost & borrow sweeps, the single-event trap, and the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`cocoa_squeeze/`](cocoa_squeeze/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
