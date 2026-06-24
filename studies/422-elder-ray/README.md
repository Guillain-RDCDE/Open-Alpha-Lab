# Study 422 — Elder Ray

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the timing edge real? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | SPY long/flat net excess Sharpe **+0.13**, HAC *t* = **+0.73** (needs ≥ 2); a rotation placebo on *when* the rule is invested gives *p* = **0.99** (random alignment does as well or better). Even **gross** *t* = +1.11. No panel tape (QQQ/IWM/EFA/GLD) clears *t* = 2 — best is GLD at **+1.77**. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | It **trails buy-and-hold by a Sharpe of −0.52** (+0.74%/yr vs +10.9%/yr), buying a milder drawdown (−39% vs −55%) with **~30 one-way round-trips/yr**. A risk-reducer, not a return engine; the long/short variant is outright negative (Sharpe −0.13). |
| **Beats a plain SMA filter?** | ![Busted](https://img.shields.io/badge/Beats_a_plain_SMA_filter%3F-Busted-8b949e?style=flat-square) | A one-line **200-day SMA filter** scores Sharpe **+0.73**, and even a bare **EMA13 cross** (Elder Ray *without* the power oscillators) scores **+0.33** — both beat the full rule's **+0.13**. The Bull/Bear Power detail subtracts value. |

> **In one sentence:** Dr Elder's Bull/Bear Power, turned into a long/flat daily timing rule, lands at SPY net Sharpe +0.13 (HAC *t* = 0.73, placebo *p* = 0.99) — it cuts drawdown but bleeds most of the market's return by sitting in cash, and a one-line 200-day moving-average filter (Sharpe +0.73) beats it on every metric, so the extra Bull/Bear Power machinery is decoration, not edge.

## What we tested

Alexander Elder's **Elder Ray** decomposes price around a 13-period EMA "consensus of value": **Bull Power = High − EMA13** and **Bear Power = Low − EMA13**. The folk rule (the second screen of his Triple Screen system): in an uptrend (EMA13 rising), buy when Bear Power is negative but *rising* (bears exhausted); step aside when the trend turns down. We take it literally as a long/flat (and long/short) daily timing book on SPY (33 years) plus a four-tape panel (QQQ, IWM, EFA, GLD), entered with **one execution lag** (position decided at the close, return earned next day) and charged realistic one-way costs × NAV. We race its **net excess-of-cash Sharpe** against buy-and-hold *and* against the two obvious simpler benchmarks — a 200-day SMA filter and a bare EMA13 cross — with a HAC *t*, a circular-rotation placebo on the position/return alignment, and a cost sweep. A deterministic synthetic tape with a *planted* trend confirms the engine recovers an edge when one exists (and that zero edge can't fake significance).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what Bull/Bear Power is, why "step aside in downtrends" trades return for drawdown, the rotation placebo in plain language, and why a one-line filter wins |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the long/flat & long/short books, HAC *t* across the panel, the rotation placebo, the head-to-head against the SMA/EMA benchmarks, the cost sweep, and a synthetic planted-trend positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`elder_ray/`](elder_ray/). Real data is Yahoo daily **total-return** bars (`auto_adjust=True`), as-of 2026-05-31. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
