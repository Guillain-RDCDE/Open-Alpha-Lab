# Study 750 — Return-to-Office 🏬

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do office REITs pop when a big employer mandates RTO? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Across ~26 dated mandates the office-REIT basket's abnormal return is a statistically-zero **−0.37%** (Welch *t* = **−0.69**, placebo *p* = **0.53**), and a *stricter* 5-day mandate moves offices **no more** than a hybrid (strict−hybrid **+0.09pp**, *t* = **0.08**). No window and no benchmark (SPY or VNQ) clears \|*t*\| ~1. **Survivorship** named here: the worst landlords (WeWork, CMBS-default towers) delisted, biasing the basket *toward* the null. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Nothing to trade: zero gross reaction, and the one-day-lag tradable variant is **−0.33%** (worse after 10 bps). The sector is a rates + secular-WFH story a single memo doesn't budge. |
| **Office rebound on RTO?** | ![Not supported](https://img.shields.io/badge/Office_rebound%3F-Not_supported-8b949e?style=flat-square) | Desks physically refilled to ~half and plateaued (Kastle proxy), and the REITs still didn't price the mandates. The "RTO saves the landlords" trade is a narrative, not a signal. |

> **In one sentence:** the vivid RTO-mandate calendar — Goldman's "aberration", Musk's "40 hours or leave", Amazon's 5-day return, the federal RTO order — moved the office-REIT basket by a statistically-zero **−0.37%** on average (*t* = −0.69, placebo *p* = 0.53), a full 5-day mandate did nothing more than a hybrid one, and a synthetic control confirms ~two dozen events can't detect any reaction of plausible size on a sector that trades on **rates and structural vacancy**, so the office rebound on RTO news is real-as-a-headline, absent-as-a-signal, and untradable.

## What we tested

Bulls on office landlords tie every rally to a back-to-work headline, so we hardcode a **transparent, cited table of ~26 real RTO-mandate announcements** by big employers — each tagged **strict** (full 5-day) or **hybrid** (2–4 day) with its date — and run a textbook short-window **event study on a sector basket**: the **cumulative abnormal return** (CAR) of an equal-weight [office-REIT basket](return_to_office/data.py) around each mandate, where "abnormal" means the basket's return minus a **market-model** fit (`basket = α + β·SPY`) estimated on a clean pre-event window. We compare strict vs hybrid CARs, add a placebo null sized to the event count, a VNQ (broad-REIT) benchmark pass, a one-day execution lag for the tradable variant, and a deterministic synthetic control with a *plantable* edge. The worst-hit landlords delisted (survivorship, named on the Signal axis); a labelled, cited [Kastle occupancy proxy](return_to_office/data.py) carries the physical RTO trend for context.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the back-to-work headlines never moved the towers, what an abnormal return is, and why offices are a rates story not a memo story — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | basket market-model CAR by strict/hybrid bucket, strict−hybrid Welch *t* + a placebo basket-window null, window & VNQ-benchmark robustness, a 1-day-lag tradable variant + costs, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`return_to_office/`](return_to_office/). Events are an explicit **hardcoded, cited table**; the priced basket is **survivor-biased** (WeWork & CMBS-default towers delisted), named on the Signal axis; the Kastle occupancy series is a **labelled proxy**, never priced. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
