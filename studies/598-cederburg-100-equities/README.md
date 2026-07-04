# Study 598 — Cederburg "100% Equities for Life" 🌍

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does 100% equities beat balanced lifecycles on wealth AND ruin? | ![Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square) | **Real vs the TDF glidepath** (mean terminal wealth HAC *t* = **+2.63**, bootstrap CI excludes 0; −13 pp cohort ruin) — seed-robust: *t* ≥ 2 in **20/20** re-draws of the simulated leg and *t* = +13.07 on the pure tape. **Not certified vs plain 60/40** for the paper's own 50/50 dom/intl blend (*t* = **+1.22**, CI straddles 0, median tie, wealth *t* flips sign across re-draws) whose ruin is *worse at the headline draw* (**7.98% vs 5.76%**; a 9/20 coin flip across re-draws). The domestic-only variant is decisively certified (*t* = +11.72) but rides US exceptionalism. Named: **85.7% of the international leg is literature-calibrated simulation** (DMS parameters, seed stated), not market data. |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Two index funds, ~5 bps, unlimited capacity — the *t* is flat across a 0→25 bps cost sweep. But the certified edge needs a **70-year horizon**, is ruined in the tape's worst retirement cohort (1929), and the ruin-risk half of the pitch never certifies. Real premium, fragile safety claim. |
| **Survives the worst cohorts (1929, 1966)?** | ![Busted](https://img.shields.io/badge/Worst_cohorts%3F-Busted-8b949e?style=flat-square) | The **1929** retiree is ruined under *every* all-equity variant (year 16–18) while 60/40 survives with 62×; the **1966** retiree survives only via the *simulated* international leg — pure-tape variants are ruined. The worst-case story rests on data that does not exist. |

> **In one sentence:** on 990 monthly-start 70-year US lifecycles, "100% equities for life" beats the target-date glidepath for real (*t* = +2.63) — but the paper's diversified 50/50 blend never certifies against a plain 60/40 (*t* = +1.22) and *raises* ruin risk on this tape (7.98% vs 5.76%), because the international half earns the DMS-calibrated ~4.3%/yr, not America's 6.9% — so the claim is **Mixed**, the vehicle **Fragile**, and its worst-cohort survival story **Busted**.

## What we tested

Anarkulova, Cederburg & O'Doherty (2023): save for 40 years, retire for 30 on the 4% rule, and hold **100% stocks (half domestic, half international) for life** — the paper says you end richer *and* run out of money less often than with a 60/40 or a target-date fund. We race `alleq` (50/50 dom/intl), `alleq_dom`, static `s6040` and a `tdf` glidepath (90→45→30% equity) over every monthly-start 70-year lifecycle on the Shiller 1871–2023 real tape, all legs CPI-deflated, 10 bps one-way + 5 bps/yr ER, weights a pure function of age (one clean lag). The international leg is honest about its limits: **market data (EFA) only from 2001-09**; before that a deterministic literature-calibrated series (DMS: 4.3%/yr geo, 17% vol, 0.60 corr, seed 598) — labeled everywhere, and every conclusion is cross-checked against a pure-tape domestic variant and a corr = 1 stress. Inference: HAC *t* with the bandwidth forced to the full 840-month overlap, an outer 600-rep circular block bootstrap of the tape, the paper's own 2,000-lifetime block bootstrap, and a 20-seed synthetic control with an exact expected-wealth null. Sibling framing: [151](../151-stocks-for-long-run/) graded the *horizon* claim; this study grades the **lifecycle allocation horse race**. As-of 2026-07-03 (tape's last complete month: 2023-06).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "100% stocks for life" actually buys you, why the TDF loses badly, why the international half drags on a US tape, and the 1929/1966 worst-case stories — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the cohort panel + HAC-840 head-to-heads, outer block-bootstrap CIs, the Cederburg 2,000-lifetime bootstrap, the international-leg sensitivity, cost sweeps, and the exact-null synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`cederburg_100_equities/`](cederburg_100_equities/). Data: Shiller monthly panel + EFA (yfinance), cache-first under `_cache/`; the pre-2001 international leg is a **literature-calibrated simulation, not market data** — named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
