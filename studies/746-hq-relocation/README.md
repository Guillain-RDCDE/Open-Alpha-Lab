# Study 746 — HQ-Relocation 🏢

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does an HQ-move announcement move the stock? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The announcement CAR is **+0.20%** at *t* = **0.21** (placebo *p* = **0.74**); the announcement *day* is **+0.18%** at *t* = **0.23**; the tax−other gap is **+0.85pp** at *t* = **0.50**. The one positive lean — a **+3.4%** quarter drift — sits at *t* = **1.34** (placebo *p* = 0.30), short of the bar. Nothing here clears *t* = 1, on 20 salience-selected survivors with a subjective tax/other label. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No announcement-day move to capture and no significant drift; the one whiff (+3.4% over a quarter, *t* = 1.34, net **+3.3%**) is un-certifiable and is what a few big-cap Texas movers in a bull market produce by construction. No sign-stable edge at any horizon. |
| **"Signal or distraction?"** | ![Non-event](https://img.shields.io/badge/Signal_or_distraction%3F-Non--event-8b949e?style=flat-square) | Both camps fail: no significant positive pop (no signal), no significant negative drift (no distraction penalty). On the tape, announcing a new headquarters is, on average, a non-event. |

> **In one sentence:** the market lore that a tax-driven HQ move (Medtronic→Ireland, Tesla / Oracle / Chevron→Texas) is either a *buy* (the tax saving gets priced in) or a *fade* (theatre hiding a weak business) is neither — across a transparent table of ~20 documented relocations the announcement CAR is **+0.20%** at *t* = 0.21, the day-one jolt is absent, the only drift is an un-certifiable **+3.4%** quarter at *t* = 1.34, and a synthetic control confirms a couple-dozen events can't detect any edge of plausible size, so it's a **non-event**, untradable at any size.

## What we tested

A tidy relocation database isn't free, so we hardcode a **transparent, labelled table** of ~20 documented **HQ relocations, 2010-2025** — inversions abroad (Eaton, Medtronic, Johnson Controls → Ireland) and the low-tax-state migration (Schwab, CBRE, HPE, Oracle, Tesla, Caterpillar, Chevron → Texas), each tagged **tax/incentive** or **other** rationale (talent/cost/proximity, e.g. GE→Boston, Boeing→Arlington) with its announcement date — and run a textbook short-window **event study**: the **cumulative abnormal return** (CAR) around each announcement, where "abnormal" means the stock's return minus a **market-model** fit (`stock = α + β·SPY`) estimated on a clean pre-event window (following [Desai & Hines 2002](docs/references.md) on inversion announcements). We test the announcement CAR, the tax−other gap, a placebo null sized to the event count, a longer post-announcement **drift** leg entered one day after the headline, and a deterministic synthetic control with a *plantable* edge. The tax/other label is subjective and the table is salience-selected — both named on the Signal axis.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why an HQ move sounds like it should matter, what an "abnormal return" is, and why the market just shrugs — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | market-model CAR by bucket, tax−other Welch *t* + a placebo non-event-window null, the announcement-day vs holdable-window split, the [+1,+63] drift + costs, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`hq_relocation/`](hq_relocation/). Events are an explicit **hardcoded, labelled table** (tax/other is the believers' framing; the tape is salience-selected & survivor-only). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
