# Study 517 — Pre-FOMC-Drift 🏦

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do stocks really drift up before FOMC meetings? | ![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | On the full 33-year SPY tape the pre-FOMC session earns **+0.227%/day** (one-sample *t* = **3.04**, Welch *t* = **2.44**, random-calendar placebo *p* = **0.007**), and **3.1% of sessions carry 14.7% of SPY's entire cumulative return**. Clears the **t ≥ 2** bar, survives the placebo, and a seed-averaged synthetic control confirms the engine is faithful. SPY is survivorship-clean. REAL. |
| **Tradability** — can you harvest it today? | ![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Costs barely dent it on liquid SPY (net *t* ≈ 2.8), but it **decayed to nothing after publication**: pre-2012 Welch *t* = **2.94**, post-2012 **0.28** (a normal day), and it is timing-sensitive (shift the window one day → *t* = 1.40). Real in the archive, dead in the present. Not INVESTABLE. |
| **"It's the run-up into 14:15"?** | ![Run-up only?: Mixed](https://img.shields.io/badge/Run--up_only%3F-Mixed-8b949e?style=flat-square) | The pre-FOMC excess splits **roughly evenly** between the overnight gap (+0.108% vs +0.030%) and the intraday session (+0.120% vs +0.005%); **neither half alone clears t = 2** — only the combined day does. The clean "all morning run-up into the release" story is only half right. |

> **In one sentence:** the pre-FOMC drift (Lucca-Moench 2015) is one of the rare calendar effects that is genuinely real — 3% of sessions carried a seventh of SPY's entire cumulative return at one-sample *t* = 3.04 — but it's a museum piece that decayed to a normal day after publication (post-2012 *t* = 0.28), and the popular "it's the intraday run-up into the 2:15 statement" framing is only half right, because a meaningful slice arrives overnight before the open.

## What we tested

We rebuild the **pre-FOMC announcement drift** as a clean calendar/event study on SPY. We
**hardcode** the scheduled-FOMC announcement calendar (261 meetings, 1994–2026, Federal Reserve),
tag the SPY trading session that ends just before each meeting (3.1% of all sessions; the meeting
date is pre-known months ahead, so no look-ahead), and compare its return to every other day with
a two-sample Welch *t*, a one-sample *t* vs zero, and a 20,000-draw **random-calendar placebo**.
The Signal axis tests the full 33-year tape; Tradability charges one-way costs × turnover and
splits the sample at **2012** to test post-publication decay (the McLean-Pontiff question); the
third axis **decomposes** the pre-FOMC return into overnight (prev close → open) vs intraday
(open → close) to test the paper's "run-up into the 2:15 release" mechanism. A deterministic,
seed-averaged synthetic control with a *planted* pre-meeting drift confirms the engine is faithful
and that zero edge cannot fake significance. SPY is survivorship-clean; the 30-name survivor basket
is kept as colour with the bias named. Distinct from [Study 67 — Fed-Drift](../67-fed-drift/) (the
original write-up, replicated here) and [Study 135 — FOMC-Cycle](../135-fomc-cycle/) (the even-week
cycle).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | how 3% of days carried a seventh of the market's entire return — and why that quietly stopped once everyone read the paper, in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the pre-FOMC vs other-day Welch *t*, the random-calendar placebo, the pre/post-2012 decay split, the overnight-vs-intraday decomposition, costs × turnover, and a seed-averaged faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run
(SPY 1993→2026, fingerprint `9687074d1493`): [docs/results.md](docs/results.md).

---

*Engine: [`pre_fomc_drift/`](pre_fomc_drift/). FOMC calendar hardcoded from Federal Reserve
schedules. Basket is **survivors** — named on the Signal axis (SPY itself is survivorship-clean).
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
