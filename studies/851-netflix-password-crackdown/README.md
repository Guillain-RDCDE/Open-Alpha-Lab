# Study 851 — Netflix Password Crackdown 🔐

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — did NFLX deliver a tradable abnormal return around the crackdown dates? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The cross-event mean event-session abnormal return (market model vs SPY) is **−5.51%** (one-sample *t* = **−0.66**, n=5), only **1/5** events up. The claimed *upside surprise* is a **single event** (Q3'23, **+17.34%** abnormal); the negative average is entirely the **2022-04 announcement crash** (−34.55%) — leave-one-out → **+1.75%**. The bootstrap CI **[−21.7%, +8.4%]** straddles zero; the QQQ cross-check agrees (−4.79%, *t* = −0.59). *Survivorship: NFLX is a single continuously-listed name — the binding limit is **N = 5 events → no power**, named on the Signal axis.* |
| **Tradability** — could you have banked it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The long-only buy-the-event overlay never reliably beats simply holding NFLX; net *t* ∈ **[−1.06, +0.34]** across 1/3/5/10/21-day holds, win-rates 20–60% on five trades. No edge to charge costs against. |
| **Upside surprise?** | ![1 of 5 events](https://img.shields.io/badge/Upside_surprise%3F-1_of_5_events-8b949e?style=flat-square) | The policy *worked* as a business fact (subscribers grew), but only **1 of 5** market-facing dates was the promised pop. |

> **In one sentence:** Netflix's 2023 "password crackdown" was a genuine corporate
> turnaround — subscribers grew and Q3'23 popped ~16% — but across the **five public
> dates** of the saga there is **no tradable abnormal-return signal** (the average is one
> crash, one pop, three shrugs), so the honest read is **a business fact, not a market
> edge — a case study, not a factor.**

## What we tested

Netflix's paid-sharing (**"password crackdown"**), feared to spike churn, became an
**upside surprise**. We run a single-name **event study** on NFLX's **abnormal returns**
(a one-factor market model vs **SPY**, cross-checked vs **QQQ**) around the **five public
market-facing dates** (2022-04-20 → 2023-10-19): the Q1'22 first flag, the 2022-08 LatAm
test, the 2023-05 broad US rollout, and the Q2/Q3'23 earnings that confirmed the
subscriber gains. Real tape: **yfinance NFLX/SPY/QQQ total-return closes, 2015 →
2026-06-30**. Method: OLS α/β on a 120-session estimation window ending 10 sessions before
the event window (strictly out-of-sample); event window [−1..+5]; earnings react next
session (one documented lag); a cross-event one-sample *t* (n=5 — **4 df, fat tails**), a
Wilson hit-rate, a **4,000-draw random-calendar placebo**, an event-bootstrap CI, a
leave-one-out cut, a costed long-only timer, and a 20-seed synthetic control (on 30
pseudo-events). **N = 5 → almost no power**, named on the **Signal** axis — a case study.
**Dedup:** [551-netflix-top10](../551-netflix-top10/) tests a *different* NFLX signal (the
Top-10 content chart), not policy-event reactions; [552-app-store-rankings](../552-app-store-rankings/)
is app-download alt-data, not a single-name event study; [299-keynote-drift](../299-keynote-drift/)
is drift around *scheduled recurring* keynotes, not one-off policy milestones; and
[622-thematic-etf-curse](../622-thematic-etf-curse/) is a fund-launch narrative, not a
stock's reaction to its own news. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "the scary policy that worked" is real as a business story but invisible as a five-event trading edge |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the market-model CARs, the fat-tailed n=5 inference, the 4,000-draw placebo, the leave-one-out cut, the costed timer, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`nflx_crackdown/`](nflx_crackdown/). NFLX/SPY/QQQ total-return closes via yfinance,
cached offline; the crackdown calendar is hardcoded from the Netflix shareholder
letters/newsroom (public record). **Not investment advice** — research & education. See
[LICENSE](../../LICENSE).*
