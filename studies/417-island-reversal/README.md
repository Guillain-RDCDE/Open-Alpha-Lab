# Study 417 — Island Reversal 🏝️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the gap-island reversal exist? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The headline figure — the **island top** (short) — has a forward excess that is **negative at every horizon** (−0.71% at 5d, −1.28% at 10d) over each name's own base rate. One-sample *t* never exceeds **−1.6** in magnitude (HAC the same), nowhere near the **t ≥ 2** bar, and the placebo beats it (random short dates do better ~52–68% of the time). No gap threshold rescues it; on SPY alone it fires **3** times in 21 years. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Net of 5 bps/leg the island-top short is **−0.81% to −1.38%** (worse than random); the island-**bottom** long's apparent *t* > 2 sits **inside the placebo cloud** (p ≥ 0.16) — pure beta, not alpha. Nothing deployable. |
| **Reliable reversal?** | ![Busted](https://img.shields.io/badge/Reliable_reversal%3F-Busted-8b949e?style=flat-square) | The two sides are asymmetric: the bearish island top **loses** (it fights the tape's up-drift) while the bullish island bottom only **rides** that drift (high *t*, but a placebo it can't beat). A real reversal would work on both sides on its own merits. This one works on neither. |

> **In one sentence:** an island reversal — a cluster of bars marooned between two opposite gaps — *looks* like a high-confidence exhaustion signal, but the closest mechanical version produces a **negative** forward excess on the bearish (island-top) side and only **drift-borrowed** returns on the bullish side that the placebo refuses, so the figure is a shape the eye finds, not a force the tape obeys.

## What we tested

We encode the closest **objective** island-reversal rule we can write down — a clear gap (≥ 1%) in one direction, a 1–3-bar island whose range never fills that first gap, then a **sealing gap** of the same size in the opposite direction that re-isolates the cluster — and run it on **SPY + 29 long-listed US large-caps** (yfinance daily auto-adjusted OHLC, 2005→2026, 21.4 years). For each confirmed island we enter **one day after** the sealing gap (no look-ahead) and measure the forward **5/10/20/40-day** return in the figure's intended direction, **net of each name's own base rate**, with a one-sample/HAC *t*, a **same-tape random-date placebo**, and one-way costs. The bearish **island top** is the headline; the bullish **island bottom** rides along as the symmetry myth-check. A deterministic synthetic control with a *planted* post-island reversal confirms the engine can bank a real edge (p = 0.000 with a planted reversal; p = 0.63 with none). Chart figures are partly subjective — we test the mechanical definition and say so. Survivorship (surviving-names basket) is named on the Signal axis and tilts *against* the headline short.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what an island reversal *looks* like, why the bearish version loses, why the bullish version only "works" because the market drifts up, and what the placebo exposes — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the mechanical detector, forward excess over base rate at 5/10/20/40d, one-sample + HAC *t*, the same-tape placebo, the gap-strictness sweep, the island-bottom symmetry myth check, SPY-only, and the synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`island_reversal/`](island_reversal/). Detector reads OHLC; forward returns on adjusted closes. Basket is **survivors** — named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
