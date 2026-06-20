# Study 322 — FOMC-Blackout 🤫

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Blackout days earn +5.00 bps/day — but so does the rest of the year (+4.73). The decisive **blackout − other** diff is **+0.27 bps, Welch *t* = +0.08**; the calendar placebo is flat and the pre/post-2011 sign flips. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A long-only-in-blackout book earns **+2.69%/yr vs +12.07%/yr** buy-and-hold, with a *lower* excess Sharpe (0.30 vs 0.64) even before costs — a fraction of beta, not an edge. |
| **Calm before the storm?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Blackout-window volatility is marginally **higher** (ratio 1.03), not lower. The quiet period in the Fed's *communications* is not a quiet period in the *market*. |

> **In one sentence:** the ~10-day pre-meeting FOMC communications blackout earns plain old equity drift and is no calmer than any other stretch of the year — the "calm before the storm" trade is just being long the index part-time.

## What we tested

Traders like to call the Fed's pre-meeting **blackout** — the ~10-day quiet period (second Saturday before a meeting through the Thursday after) when officials may not speak publicly — a *"calm before the storm"*: a steady, positive drift while the market waits, with the fireworks saved for the decision itself. We flag every SPY trading day since 1994 as inside or outside that pre-meeting window (from the public [FOMC schedule](fomc_blackout/data.py)) and ask the only two questions that matter: does the blackout window earn an *excess* return over the rest of the year, and is it actually calmer? We pin it with a calendar-shift placebo, a pre/post-2011 decay split, an excess-of-cash Sharpe race against buy-and-hold, and a deterministic synthetic positive control. This is a deliberately *different angle* from [Study 135 — FOMC-Cycle](../../135-fomc-cycle/) (the Cieslak even/odd-week *post*-meeting drift): here it is the **pre-meeting blackout window** itself.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the blackout in plain language, why "+5 bps/day" is a trick, the calm myth, and what a part-time index bet actually earns |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | blackout-vs-rest HAC/Welch t, the calendar placebo, pre/post-2011, the vol check, the excess-Sharpe race, and the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`fomc_blackout/`](fomc_blackout/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
