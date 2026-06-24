# Study 434 — DEMA & TEMA 📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there a real timing edge? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | No moving-average line beats buy-and-hold on 21 years of SPY. The daily *excess-over-buy-and-hold* HAC *t* is **negative** for all of them (SMA −2.30, EMA −1.86, DEMA −2.79, **TEMA −3.49**). Nothing clears the **+2** bar. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | TEMA nets a Sharpe of **+0.24** < the plain SMA's **+0.54** < buy-and-hold **+0.65**, while trading **~2× more** (39 vs 17 round-trips/yr). Its *timing* even loses to a random schedule of the same days (placebo *p* = **0.88**). |
| **"Worth the extra smoothing?"** | ![Busted](https://img.shields.io/badge/Worth_it%3F-Busted-8b949e?style=flat-square) | The lag-cancellation algebra is real, but **TEMA − SMA runs −3.0%/yr** (HAC *t* = −1.25) and DEMA − SMA −1.2%/yr (−0.64). Less lag bought a strictly *worse* rule. |

> **In one sentence:** DEMA and TEMA really do cut a moving average's lag — and on 21 years of daily SPY that buys you a *worse* trend rule: turned into the same long/flat signal, TEMA nets Sharpe +0.24 vs the plain SMA's +0.54 and buy-and-hold's +0.65, at twice the turnover, with every outperformance HAC *t* negative and its timing beaten by a random schedule 88% of the time — because less lag means more reaction to noise.

## What we tested

We hold the timing logic fixed — **long when the close is above its moving average, flat otherwise**, at a **50-day window**, with one documented execution lag (signal at the close of *t*, return of *t+1*) and one-way costs × turnover — and swap **only the line**: a plain SMA, an EMA, and the "less-lag" DEMA (Mulloy, 1994) and TEMA. We then race the **net, excess-of-cash Sharpe** against two benchmarks the believer must beat: **buy-and-hold** (does any trend rule even help?) and the **plain SMA** (does the improved line beat the boring one it claims to upgrade?). The Signal axis uses a Newey-West HAC *t* on the daily excess-over-buy-and-hold return plus a circular block-permutation null that randomises the timing while holding the activity level fixed; Tradability sweeps costs to 10 bps. A deterministic synthetic control with a *planted* trend confirms the harness banks a real edge (SMA Sharpe to +4.0) — and that DEMA/TEMA whipsaw even then.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "lag" means, why DEMA/TEMA hug price tighter, and why that tighter hug *loses* money once you trade it — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the net excess-of-cash Sharpe race across SMA/EMA/DEMA/TEMA, HAC *t* on excess-over-B&H, a paired X-minus-SMA test, a block-permutation timing null, the cost sweep, and a planted-trend synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`dema_tema/`](dema_tema/). Real tape is SPY daily 2005-2026, Yahoo `auto_adjust=True` (total-return-ish). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
