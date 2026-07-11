# Study 647 — PBoC RRR Effect 🇨🇳

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do Chinese equities really pop when the PBoC cuts the RRR? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | No individual split clears *t* ≥ 2 (cut days Welch *t* = **0.99**, hike days *t* = **1.36**, both n < 32), and the decisive direct test — **cuts vs hikes, Welch *t* = +0.09 on both FXI and MCHI** — shows the folklore's own prediction fails: hike days run just as hot (+0.64%) as cut days (+0.71%). RRR announcement days ARE louder than average (range *t* = +2.45) — the PBoC just times them to eventful stretches; direction carries no reliable sign. |
| **Tradability** — can you "buy the cut"? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | A prior-close-to-close timer at 1/3/5/10-day holds never reaches *t* = 2 (best: *t* = 0.95); the 10-day hold shows **zero** excess over a random window of the same length (*t* = −0.00) with a **−29.8%** worst-case single event, and FXI's own swap-heavy QFII-quota structure adds tracking fragility on top. |
| **"Buy the rumor, sell the news"?** | ![Busted](https://img.shields.io/badge/Buy_the_rumor%2C_sell_the_news%3F-Busted-8b949e?style=flat-square) | No rumor phase (pre-cut run-up is *negative*, *t* = −0.97) and no clean news-selloff (post-cut window flat/negative, *t* = −0.71) — the mechanism isn't there to debunk a subtler way; it's just not present. |

> **In one sentence:** across 48 broad-based PBoC RRR announcements since 2008 — 31 cuts, 17
> hikes — Chinese-equity ETFs (FXI, cross-checked on MCHI) run hot on ANY RRR announcement day
> (+0.6-0.7% vs ≈ 0% normally) but **cut days and hike days are statistically indistinguishable
> from each other** (Welch *t* = 0.09), no split individually clears the desk bar, and no
> holding period of a "buy the cut" timer nets a certifiable edge after costs — the "stimulus
> rally" is a **`None`** on direction and a **`Mirage`** to trade.

## What we tested

Chinese-equity folklore says a PBoC RRR cut is a stimulus green light and equities should pop —
implicitly, a hike should do the opposite. We hardcode **48 broad-based** (system-wide, large-
financial-institutions) PBoC RRR announcements 2008-01-16 → 2025-05-07 — 31 cuts, 17 hikes,
targeted/structural relief for rural or SME-focused lenders excluded — and run FXI's (China
large-cap ETF) log return on each announcement's mapped trading day vs every other day: Welch
*t*, Newey-West, a one-sided random-calendar placebo, an event window [−5..+10] for run-up and
persistence, and — the decisive test — a direct cuts-vs-hikes Welch *t*. MCHI (a differently
constructed China ETF, 2011 inception onward) cross-checks every number. The third axis asks the
honest timing question the brief invites: does the tape at least reward *buying the rumor and
selling the news* around a cut, even if the average is flat? **Dedup:**
[620-a-h-premium](../620-a-h-premium/) is a structural cross-sectional price gap with no event
calendar (no overlap); [313-geopolitical-shock](../313-geopolitical-shock/) shares the
event-study/placebo/synthetic-control skeleton but tests a completely different market and
calendar (wars on SPY, not RRR moves on China). As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the RRR actually does, why "stimulus = pop" sounds obviously true, and the one clean test (cuts vs hikes) that quietly kills it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the Welch/HAC splits, the one-sided placebo, the [−5..+10] event anatomy, the MCHI cross-check, the era contrast, the buy-the-cut timer with costs, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`pboc_rrr_effect/`](pboc_rrr_effect/). The RRR calendar is hardcoded from PBoC
official announcements cross-checked against financial-press coverage; FXI/MCHI are funds, not
survivor-conditioned baskets — FXI's swap-heavy QFII-quota structure is named on the
Tradability axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
