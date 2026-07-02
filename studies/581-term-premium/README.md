# Study 581 — Term-Premium

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a model term-premium estimate time long-duration returns? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | On the 21-day horizon the fattest-premium quintile beat the thinnest by **+0.26%** — the *right* direction — but HAC *t* = **+0.43** (placebo *p* 0.72), far below the bar, right-signed at every horizon yet never clearing *t* = 2, and it **flips sign** across sub-periods (real only in 2002-09, *t* +2.09; inverts in 2017-21). Literature says real; this ACM-*proxy* tape can't certify it. |
| **Tradability** — does the timing overlay pay? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Owning TLT on fat-premium days edges buy-and-hold on Sharpe (**0.291 vs 0.247**) only by sitting in cash 57% of the time — its *mean-return* spread is **−0.32 bps/day** (*t* −0.39) at ~9.6 switches/year. Volatility-reduction artefact, not alpha. |
| **"Term-premium times duration?"** | ![Mixed](https://img.shields.io/badge/Mixed-8b949e?style=flat-square) | It **does** time duration in 2002-2009 (*t* +2.09) but fades to zero in 2010-16 and **inverts** in the ZIRP 2017-21 era (*t* −0.82). Present in 1 of 4 windows, gone or backwards in the rest. |

> **In one sentence:** an ACM-style term-premium estimate — the 10-year yield stripped of an expectations component — points the *right way* at forward long-bond (TLT) returns, but on 2002-2026 daily data the edge is faint (HAC *t* +0.43), sign-unstable across regimes, and untradable (the timer ties buy-and-hold and loses on mean return after ~10 switches a year).

## What we tested

The **term premium** (Adrian-Crump-Moench 2013; Cochrane-Piazzesi 2005; Fama-Bliss 1987) is the
part of a long bond's yield *not* explained by expected future short rates — the time-varying
compensation for duration risk. The claim: when the premium is fat, long bonds are richly paid and
should out-earn cash going forward, so a model term-premium estimate ought to **time long-duration
returns**. Because the NY-Fed ACM series isn't reachable from a no-key retail stack, we build the
cheapest honest proxy — `tp = 10-year yield − EWMA(short rate)`, which strips the expectations
component and is exactly what separates a term-*premium* from the raw curve *slope*
([Study 132](../132-yield-curve-steepener/)) — rank it out-of-sample, and test whether the
fattest-premium quintile precedes higher forward TLT returns. Tests: a Q5−Q1 forward-return sort
with a **HAC (Newey-West) *t***, a **block-shuffle placebo** null, a horizon sweep, a four-window
sub-period sweep (the sign-stability check), a costed timing overlay vs buy-and-hold TLT, and a
deterministic, seed-robust **synthetic positive control** that plants a timing edge and proves the
engine catches it. *A model-**proxy** study: `REAL` (which needs a robust real-tape *t* ≥ 2) is out
of reach in principle here — the honest ceiling is `WEAK`.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a term premium is, why it *should* predict long-bond returns, and why the edge here is real-but-tiny and regime-dependent |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the quintile sort with a HAC *t*, the placebo null, the horizon & sub-period sweeps, the timing-overlay Sharpe mirage, and the seed-robust synthetic positive control |

The fingerprinted real-data run (TLT + `^TNX` + `^IRX`, 2002-07-31 → 2026-06-26, 6,009 days, tape fp
`83b8f0823f11`) is in [docs/results.md](docs/results.md); the offline machinery proof runs on the
deterministic synthetic world in [`term_premium/data.py`](term_premium/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`term_premium/`](term_premium/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
