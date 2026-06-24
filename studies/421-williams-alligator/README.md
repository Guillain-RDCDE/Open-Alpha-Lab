# Study 421 — Williams Alligator 🐊

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there a timing edge? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The long/flat Alligator's Sharpe (**+0.593**) edges buy-and-hold (**+0.551**) only by sitting in cash 41% of the time — its **mean return is lower**. The Sharpe-difference HAC *t* is **−1.71** (below the |*t*| ≥ 2 bar), and a 21-day block-shuffle placebo beats B&H as much as the real signal **15%** of the time (*p* = 0.151). Long/short is weaker (Sharpe +0.156, *t* = +1.05). No timing edge survives. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Net of 5 bps one-way costs the Alligator earns **+6.46%** CAGR vs buy-and-hold's **+10.78%**, and is **strictly dominated by a one-line SMA(200)** (Sharpe +0.726 vs +0.593; Sharpe-diff *t* = −1.61). Even *gross* it loses to the simpler rule. Three displaced averages buy nothing one average doesn't do better. |
| **Does the waking alligator catch trends?** | ![Not_supported](https://img.shields.io/badge/Catches_trends%3F-Not_supported-8b949e?style=flat-square) | It catches *crashes* (by being out on downtrends), not *trends* (by being in at the right time). The placebo and a faithful synthetic positive control (which detects planted trends at *t* > 4) together show the only real effect is **reduced exposure**, not the "wake and eat" timing the folklore sells. |

> **In one sentence:** Bill Williams' famous three-line Alligator, run as a long/flat (and long/short) timing rule on 33 years of SPY, is a volatility-reducing crash filter whose tiny Sharpe edge over buy-and-hold doesn't clear the inference bar (Sharpe-diff *t* = −1.71, placebo *p* = 0.15) — and it is **beaten outright by the single dumbest trend rule there is, a 200-day moving average**, so the "catches trends better" claim simply isn't what the tape shows.

## What we tested

We compute the canonical Alligator — three smoothed moving averages (SMMA / Wilder) of the daily **median price**, displaced forward in time: **Jaw** (13-period, +8 bars), **Teeth** (8, +5), **Lips** (5, +3). When the lines fan out in order (**Lips > Teeth > Jaw**) the "alligator wakes" and we go long; long/flat goes to cash otherwise, long/short shorts the bearish fan. We race the **NET** Sharpe (excess-of-cash vs excess-of-cash, 5 bps one-way on NAV, 50 bps/yr borrow on shorts, one documented execution lag) against buy-and-hold **and** against the simplest possible trend benchmark, **SMA(200)**, on SPY daily total-return closes 1993→2026. The Signal axis uses a HAC (Newey-West) Sharpe-difference *t* and a circular-block permutation placebo; a deterministic synthetic tape with *planted* multi-week trends confirms the engine catches trends when they exist and raises no false alarm on an i.i.d. null (a machinery proof only — it never backs the real stamp).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the three lines are, what "wake and eat" means, the three-way scorecard, and why a single moving average beats the whole apparatus — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the displaced-SMMA fan, long/flat & long/short backtests, HAC Sharpe-difference races vs B&H and SMA(200), a 2,000-draw block-permutation placebo, a cost sweep, and a faithful-engine synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`williams_alligator/`](williams_alligator/). Indicator = three displaced SMMAs of the median price (Williams' 13/8/5 fan). Benchmark = SMA(200) through the identical engine. Cash leg proxied at 4%/yr flat (FRED unavailable). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
