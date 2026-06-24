# Study 432 — Hull Moving Average

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does HMA timing beat holding? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The HMA(16)-slope long/flat rule's daily **active spread vs buy-and-hold is −4.35 bps/day at HAC *t* = −5.31** on SPY (gross *t* = −5.01), negative and significant on **all five** tapes and **both** sample halves. A position-shuffle permutation gives ***p* = 1.000** — the timing is *worse* than random. The lag-free HMA adds **negative** timing information. |
| **Tradability** — could you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Net Sharpe **0.089 vs buy-and-hold 0.646**; even **gross** it trails. ~32 switches/yr means costs only deepen the hole (no break-even cost exists), and the long/short version is outright destructive (Sharpe −0.55). |
| **"Cuts false signals?"** | ![Busted](https://img.shields.io/badge/Cuts_false_signals%3F-Busted-8b949e?style=flat-square) | The HMA fires **32.5 position changes/yr vs the SMA's 17.4** — *nearly double* the whipsaws — and **underperforms that simpler SMA rule** (Sharpe 0.089 vs 0.465). Lower lag bought *more* false signals, not fewer. The headline claim is reversed on the tape. |

> **In one sentence:** the Hull Moving Average really does lag less than an SMA — but on daily US equities that lower lag buys *more* whipsaws (32.5 vs 17.4 switches/yr), not fewer, so its long/flat timing rule trails plain buy-and-hold by 4.35 bps/day at *t* = −5.31, loses to the simpler SMA it's meant to beat, and a permutation placebo (*p* = 1.0) shows the timing carries negative information — while a synthetic tape with a planted trend confirms the engine banks a real trend the moment one exists.

## What we tested

A staple of TradingView scripts and broker tutorials: *"The Hull Moving Average hugs price with far less lag than an SMA or EMA, so a trend rule built on it turns earlier and produces **fewer false signals** — it beats a plain moving average."* We take it literally: compute Alan Hull's `HMA(n) = WMA(2·WMA(n/2) − WMA(n), √n)`, turn its slope into a daily **long/flat** (and, separately, **long/short**) timing rule with one documented execution lag (position formed on the close of *t* earns *t+1*'s return), and race it **net of one-way costs × NAV** against **buy-and-hold** and against the obvious simpler benchmarks — an **SMA(50)** crossover and **MACD** — on SPY, QQQ, AAPL, MSFT and XLE (total-return daily bars, full history to 2026-06-12). The Signal axis tests the daily *active spread* (strategy − buy&hold) with a Newey-West HAC *t* and a 2,000-draw position-shuffle permutation; the third axis counts whipsaws head-to-head against the SMA. A deterministic synthetic tape with a *planted* trend is the positive control proving the harness banks an edge when one is there.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the HMA is and why it lags less, why "fewer false signals" is the opposite of true on noisy daily bars, why timing must be raced against just holding, in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HMA vs SMA vs MACD vs buy&hold, the active-spread HAC *t*, the whipsaw count, the position-shuffle permutation, cost & per-instrument & in/out-of-sample sweeps, and the planted-trend positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`hull_moving_average/`](hull_moving_average/). Daily bars are **total-return** (`auto_adjust=True`); all Sharpe figures are net and excess-of-cash. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
