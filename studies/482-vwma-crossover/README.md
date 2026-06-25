# Study 482 — VWMA-Crossover 📊

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the VWMA cross forecast? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The VWMA golden cross does **not** beat a drift-matched **random-entry** baseline: VWMA − random = **−12.5 / +7.1 / +47.3 / −31.6 bps** at 5/10/20/60 days, and the VWMA-vs-random Welch *t* **never clears 2** (max **+1.45** at 20d, *p* = 0.15). The big one-sample *t*'s (20d **+4.81**, 60d **+5.31**) are **pure beta** — the upward drift every golden cross inherits. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No residual edge once the free drift is removed; costs only deepen the hole. You'd capture the same drift more cheaply by **holding the index**. Nothing to scale. |
| **"Does volume-weighting add edge?"** | ![Busted](https://img.shields.io/badge/Volume--weighting_adds_edge%3F-Busted-8b949e?style=flat-square) | VWMA − SMA (identical lengths, only the weighting differs) is **negative at every horizon** (−9.8 / −17.0 / −12.4 / −36.6 bps; Welch *t* negative everywhere), the weighting helps in only **1 of 5** names, and scrambling the volume leaves the result intact (**p = 0.24**). The volume term carries no information. |

> **In one sentence:** A volume-weighted moving-average cross *sounds* smarter than a plain one — but run them **head-to-head** at identical lengths across 5 indices over 21 years and the volume-weighted golden cross **loses to the plain SMA cross at every horizon** (and to buying on random days), with the shuffled-volume placebo leaving the result untouched (*p* = 0.24): the volume term is dead weight, and the apparent profit is all market drift.

## What we tested

We encode the tightest mechanical version a proponent would accept. The **VWMA** is the trailing volume-weighted mean `Σ(price·vol)/Σ(vol)`; a long fires when the fast VWMA (10) crosses **above** the slow VWMA (30), entered at the **next close** (one documented lag), and we measure the forward 5/10/20/60-day return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return **+ volume**, 2005→2026). The thesis test — *does volume-weighting add edge?* — is the **head-to-head VWMA − SMA** Welch *t*, holding fast/slow lengths, the cross rule, the instrument and the hold fixed so the **only** difference is the weighting. The Signal axis is **VWMA vs a drift-matched random-entry baseline** (the only honest test on an upward-drifting tape), plus a **shuffled-volume placebo** that destroys the weighting while keeping the price path and the volume marginal. Tradability charges costs on every cross. A deterministic synthetic control with a *planted* volume-led drift pulse proves the detector is live (edge 0 → VWMA−SMA ≈ 0; planted pulse → VWMA cross *t* = +7.17, VWMA−SMA = +53 bps), so the flat real-tape result is a genuine "nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a VWMA is, why a golden cross on a rising market always looks good, the VWMA-vs-SMA race, and the volume scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | causal VWMA/SMA crosses, one-sample HAC *t* vs the beta trap, the head-to-head VWMA−SMA Welch test, the random-entry Signal test, the shuffled-volume placebo, per-ticker deltas, costs, and a synthetic planted-volume control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`vwma_crossover/`](vwma_crossover/). Moving averages are causal (trailing windows only); the cross is read on close of *t*, entry is the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument trend study, so the random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
