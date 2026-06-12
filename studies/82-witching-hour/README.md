# Study 82 — Witching-Hour

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | Volume is `REAL`: +24% on expiry day (HAC *t* = +5.72, year-demeaned). Range is `NONE` (*t* = −1.39). Return is `REAL` but **negative**: −19 bps differential on expiry Friday (HAC *t* = −2.34). |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Long-witching-week strategy earns Sharpe +0.52, HAC *t* = +0.88 — the equity premium, not alpha. The negative Friday return is too thin to short against the equity headwind. |
| **Mechanical effect?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | The Stoll-Whaley (1987/1991) expiry-day volume effect survives in modern SPY data. What died after 1988 is the price-dislocation at settlement — the volume surge did not. |

> **In one sentence:** triple/quadruple witching reliably pushes +24% more volume through SPY (REAL, *t* = 5.7) but does *not* elevate intraday volatility — the flows are predictable roll trades, not informed orders — and any directional return edge is a modest *negative* bias on expiry Friday only (*t* = −2.34), too thin to overcome execution costs or the equity premium headwind.

## What we tested

Every quarter-end the desk receives the same question: *"Is witching week different — higher vol, higher volume, and a predictable direction?"* The folk version: options and futures expiry forces huge, calendar-known hedging and roll flows that elevate volume and volatility, and the net selling pressure pushes the market down on expiry Friday. We test each claim separately on SPY daily data from 1993 (8,400 trading days, 136 witching events) using Newey-West HAC t-stats and a year-demeaned volume correction that strips the secular SPY volume decline — without it, the naïve log-volume *t*-stat is +0.99 (noise); with it, +5.72.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the three claims in plain language, why volume spikes but price doesn't, the negative-return paradox, and why you can't trade it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stats, year-demeaned volume maths, the Kyle (1985) paradox, the synthetic positive control, and the cost sweep on the Friday-short |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`witching_hour/`](witching_hour/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
