# Study 484 — Vertical-Horizontal-Filter 📐

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the VHF gate help? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Gating a momentum entry on a high VHF does **not** beat the **same entry ungated**: gate − ungated = **−2.1 / −9.7 / −5.9 / +5.8 bps** at 5/10/20/60 days, and the gate-vs-ungated Welch *t* **never clears 2** (range **−0.60 to +0.13**). It also ties a drift-matched random baseline (Welch *t* −0.51 to +0.68). The big one-sample *t*'s (20d **+3.35**, 60d **+5.97**) are **pure beta** — the ungated momentum entry already has them. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No residual edge once you compare against the cheaper ungated entry. The gate merely **discards ~⅓ of the momentum trades** (608 gated vs 920 ungated) for no compensating gain; costs only deepen the hole. You'd capture the same drift more cheaply by **holding the index**. |
| **"Does the VHF gate add edge?"** | ![Busted](https://img.shields.io/badge/VHF_gate_adds_edge%3F-Busted-8b949e?style=flat-square) | Scramble *when* the gate fires (shuffled-gate timing placebo) and the result barely moves: **p = 0.100** of time-scrambled gates match or beat the real one, and the gate-minus-ungated delta **flips sign** across the five names (+68 SPY, −89 GLD). The VHF's "trending now" timing carries no information. |

> **In one sentence:** The Vertical-Horizontal-Filter looks useful because it lights up exactly when an upward-drifting index has been rising — so a VHF-gated momentum entry inherits the same drift the plain momentum entry already had, and across 5 indices over 21 years the gate **loses to the ungated entry** at 5–20 days (gate-vs-ungated *t* never clears 2; the timing placebo leaves it intact, *p* = 0.10): all tide, no tool.

## What we tested

We encode the tightest mechanical version a proponent would accept. The momentum trigger is a **50-day moving-average cross** (close above its 50d MA, read on the close of *t*). The **VHF** (White, 1991) is `|highest − lowest| / Σ|close diff|` over a **28-day** window; the gate keeps only the momentum entries whose VHF sits in the **top tertile** of its **trailing 252-day** distribution ("the VHF says trending"), entered at the **next close** (one documented lag), measured at forward 5/10/20/60 days on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **gated vs ungated momentum** — a strictly drift-matched comparison, since both sides ride the same index — plus a **drift-matched random baseline** and a **shuffled-gate timing placebo** that decouples the "trending" label from price while keeping its marginal. Tradability charges costs on every gated entry. A deterministic synthetic control with a *planted* VHF-conditional regime proves the detector is live (edge 0 → gate ≈ ungated, Δ = −20 bps; planted regime → gate beats ungated by **+377 bps**, *t* = +6.05), so the flat real-tape result is a genuine "nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the VHF is, why a momentum gate on a rising market always looks good, the gated-vs-ungated race, and the gate-timing scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the VHF gate, one-sample HAC *t* vs the beta trap, the ungated-momentum Welch test, the shuffled-gate placebo, per-ticker deltas, costs, and a synthetic planted-regime control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`vertical_horizontal_filter/`](vertical_horizontal_filter/). Momentum = close > 50d MA; VHF window 28; gate = VHF in the trailing-252d top tertile; entry is the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument trend study, and the decisive test is gated-vs-ungated (both ride the same drift), so survivorship affects only the neutralized baseline. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
