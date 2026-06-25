# Study 469 — Relative Vigor Index 📊

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the RVI cross channel price? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The "buy the RVI cross-up" rule does **not** beat a drift-matched **random-entry** baseline: cross − random = **−5.0 / −12.6 / −5.4 / +11.9 bps** at 5/10/20/60 days, and the cross-vs-random Welch *t* **never clears 2** (max **+0.49** at 60d, *p* = 0.621). The big one-sample *t*'s (20d **+7.39**, 60d **+8.36**) are **pure beta** — the upward drift every long-only entry inherits. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No residual edge once the free drift is removed; costs only deepen the hole. You'd capture the same drift more cheaply by **holding the index**. Nothing to scale. |
| **"Does the RVI cross forecast?"** | ![Busted](https://img.shields.io/badge/RVI_cross_forecasts%3F-Busted-8b949e?style=flat-square) | Slide the indicator out of phase with price (phase-scramble placebo) and the result barely moves: **56%** of scrambled crosses match or beat the real one (*p* = **0.557**). The specific RVI/signal crossover carries no information. |

> **In one sentence:** the Relative Vigor Index cross-up looks promising because indices drift up — encode it mechanically (Ehlers' causal 4-bar smoother, signal line, no eyeballing) and fire the "buy the cross" rule 2,427 times across 5 indices over 21 years, and it **loses to buying on random days** at 5–20 days (and the timing placebo leaves the result untouched, *p* = 0.56): all tide, no tool.

## What we tested

We encode the tightest mechanical version a proponent would accept. The RVI is Ehlers' causal momentum oscillator — a 4-bar symmetric-weighted smoother (weights 1,2,2,1) applied to the bar **body** (close − open) and **range** (high − low), summed over **N = 10** bars; the **signal line** is the same smoother applied to the RVI. A long fires when the RVI crosses **from below to above** its signal line (read on the close of *t*), entered at the **next close** (one documented lag), and we measure the forward 5/10/20/60-day return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **cross vs a drift-matched random-entry baseline** (a Welch *t*) — the only honest test on an upward-drifting tape — plus a **phase-scramble placebo** that slides the indicator out of phase with price while keeping every RVI value. Tradability charges costs on every cross. A deterministic synthetic control with a *planted* persistent regime proves the detector is live (edge 0 → *t* ≈ 0; planted regime → *t* = +2.88), so the flat real-tape result is a genuine "nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the RVI is, why a long-only entry on a rising market always looks good, the cross-vs-random race, and the timing scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the mechanical RVI, one-sample HAC *t* vs the beta trap, the random-entry Welch test, the phase-scramble placebo, per-ticker deltas, costs, and a synthetic planted-regime control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`relative_vigor_index/`](relative_vigor_index/). The RVI and signal line are strictly causal (Ehlers' 4-bar SWMA, 10-bar sum); entry is the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument momentum study, so the random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
