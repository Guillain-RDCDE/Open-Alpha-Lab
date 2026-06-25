# Study 488 — FRAMA (Fractal Adaptive Moving Average) 〰️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the FRAMA cross-up forecast? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The "buy when price crosses above FRAMA" rule does **not** beat a drift-matched **random-entry** baseline: FRAMA − random = **−10.8 / +0.4 / +2.8 / +27.1 bps** at 5/10/20/60 days, a dead heat, and the FRAMA-vs-random Welch *t* **never clears 2** (max **+1.17** at 60d, *p* = 0.244; *negative* at 5d). The big one-sample *t*'s (20d **+7.27**, 60d **+8.08**) are **pure beta** — the upward drift every long-only entry inherits. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No residual edge once the free drift is removed — and a **plain fixed EMA** of the same speed matches FRAMA (Δ_ema = **+0.0 / +5.0 / +13.4 / +6.3 bps**), so the *adaptive* part buys nothing. You'd capture the same drift more cheaply by **holding the index**. Nothing to scale. |
| **"Does fractal-adaptive smoothing buy edge?"** | ![Busted](https://img.shields.io/badge/Adaptive_edge%3F-Busted-8b949e?style=flat-square) | Time-scramble FRAMA's adaptive smoothing (shuffled-alpha placebo) and the result barely moves: **42%** of scrambled-adaptation runs match or beat the real one (*p* = **0.417**). The fractal dimension carries no information. |

> **In one sentence:** FRAMA looks smart because it changes speed and because indices drift up — encode it mechanically (Ehlers' exact recursion, no eyeballing) and fire the "cross above FRAMA" rule 2,832 times across 5 indices over 21 years, and it **ties buying on random days** and **ties a plain fixed EMA** (and scrambling the fractal adaptation leaves it untouched, *p* = 0.42): all tide, no tool.

## What we tested

We encode FRAMA exactly as John Ehlers specifies (*Stocks & Commodities*, 2005): over an N = 16-bar window we estimate the **fractal dimension** `D ∈ [1,2]` from the two-halves price ranges, set the adaptive smoothing `alpha = exp(−4.6·(D−1))`, and run the strictly-causal recursion `FRAMA_t = alpha·C_t + (1−alpha)·FRAMA_{t−1}` (fast in trends, slow in chop). A long fires the bar the close first **crosses above FRAMA**, entered at the **next close** (one documented lag), and we measure the forward 5/10/20/60-day return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **cross-up vs a drift-matched random-entry baseline** (a Welch *t*) — the only honest test on an upward-drifting tape — plus a **fixed-EMA comparator** (same rule, one speed: does the *adaptation* add anything?) and a **shuffled-alpha placebo** that re-times the adaptive smoothing while keeping its marginal. Tradability charges costs on every cross. A deterministic synthetic control with a *planted* persistent trend proves the detector is live (edge 0 → *t* = +0.12; planted trend → *t* = +4.11), so the flat real-tape result is a genuine "nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what FRAMA is, why a "buy above the average" rule on a rising market always looks good, the cross-up-vs-random race, FRAMA vs a plain EMA, and the adaptation scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | causal FRAMA, one-sample HAC *t* vs the beta trap, the random-entry Welch test, the fixed-EMA comparator, the shuffled-alpha placebo, per-ticker deltas, costs, and a synthetic planted-trend control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`frama/`](frama/). FRAMA is a strictly causal recursion (N = 16, rolling fractal dimension, no look-ahead); entry is the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument trend study, so the random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
