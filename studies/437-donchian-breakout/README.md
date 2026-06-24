# Study 437 — Donchian-Breakout 📈

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the 20-day channel carry timing skill? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | On SPY (1993–2026, n=8,400) the block-permutation placebo gives **p = 0.83** — random in/out schedules with the same exposure beat the real breakout **83%** of the time. It *loses* to buy-and-hold on a risk-adjusted basis (HAC *t* = **−2.32**) and is statistically tied with a matched random control (*t* = +0.66). Its own +2.90 *t* is bull-market **beta** from being long 62% of the time, not breakout skill. |
| **Tradability** — could you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Even **gross** it trails buy-and-hold (Sharpe **+0.477 vs +0.553**); the long/short variant goes **negative** (−0.015); it fails across the whole panel (QQQ, TLT, GLD, UUP). The only thing it delivers — lower drawdown (−42% vs −55%) — is just lower equity exposure you can buy cheaper with a static stock/cash split. |
| **Better than a plain SMA?** | ![Not_supported](https://img.shields.io/badge/Better_than_SMA%3F-Not_supported-8b949e?style=flat-square) | Same window, same lag, same costs: the Donchian channel and an **SMA(20) cross** are statistically indistinguishable (*t* = **+0.94**) — and both lose to buy-and-hold. The channel's extra state machine buys nothing over the simplest moving-average filter. |

> **In one sentence:** stripped of the rest of the Turtle apparatus, the 20-day Donchian breakout on SPY carries no timing information (permutation *p* = 0.83), loses to buy-and-hold net *and gross* (HAC *t* = −2.32), and is a statistical tie with a plain SMA(20) — its only honest output is lower drawdown, which is just the lower beta of sitting in cash 38% of the time.

## What we tested

A staple of trend-following lore, and the entry skeleton of the legendary Turtle system: *"Buy when the close makes a new 20-day high; go flat (or short) when it makes a new 20-day low — ride the channel."* We take the **20-day channel on its own**, sans the Turtle's ATR sizing, pyramiding and opposite-channel exit, and turn it into a long/flat (and long/short) daily timing rule on SPY total-return closes back to 1993 (plus a 7-ETF panel). We race it **NET** (one-way costs × NAV, one documented execution lag, shorts pay borrow), excess-vs-excess, against buy-and-hold, a matched **random-timing control**, and the obvious simpler benchmark — an **SMA(20) cross** — so the "it's better" claim is actually tested. A block-permutation placebo asks whether the *timing* (which days) carries any information; a deterministic synthetic tape with a **planted trend-persistence knob** confirms the harness can bank a real breakout edge (it lights up at *t* = +2.24 when one is planted) and that zero edge cannot fake significance.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a Donchian channel is, why "new 20-day high = buy" feels right, why beating buy-and-hold is the real bar, why lower drawdown isn't free, and why it ties a plain moving average — in plain words |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the four-arm net race, HAC return-difference *t*-stats (vs BH / SMA / random), the block-permutation placebo, cost & window sweeps, the long/short blow-up, the panel, and the planted-edge synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`donchian_breakout/`](donchian_breakout/). SPY is **price = total-return** (`auto_adjust=True`); this is a **surviving, liquid** instrument set — named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
