# Study 960 — The Unstaked ETF ⛓️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the edge real? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | Two legs, two answers. The fund-vs-coin gap the question asks for is **unmeasurable**: ETHA tracks ETH-USD at **+0.27 %/yr** with a bootstrap CI of **[−8.9, +9.3]** (monthly *t* = +0.11) — and it is unmeasurable *in principle*, since ETH-USD is itself an **unstaked** price from which the reward is missing by construction. What the tape *does* resolve is real: the **ETHE−ETHA** spread is **−1.64 %/yr**, CI **[−2.27, −0.97]**, monthly *t* = **−5.16**, 22/24 months negative, interval clear of zero at every block length — 73% of a documented 2.25 %/yr fee gap. Caveats we do not bury: the plain i.i.d. daily *t* is only **−1.32** (we discard it, and say why — a −0.47 one-day reversal it assumes away), and this is **two of the cohort's nine funds, at its widest fee gap**. A fee measurement, not a staking one. |
| **Tradability** — is it bankable? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Owning the cheap wrapper instead of the dear one is worth the **measured 1.64 %/yr** — 3.1% of terminal wealth over 1.9 years — for one trade and no borrow, against a *documented* 2.25 %/yr fee gap. Genuinely bankable, but on 23 months of one asset and one fee-extreme pair, and the spread has already narrowed from −2.43 to −0.90 %/yr. The levered version (long ETHA / short ETHE) dies above ~145 bps of borrow. The ~3 %/yr staking prize cannot be banked inside any wrapper tested here. |

> **In one sentence:** The forgone staking yield is real arithmetic and **92% of the cheap wrapper's declared cost of ownership** — but it is arithmetic on two external constants, because the coin benchmark is unstaked and this tape cannot resolve anything smaller than **±5 to ±18 %/yr** depending on which ruler you hold (every one of them larger than the ~3 %/yr under discussion); point the same estimator at two funds sharing one closing bell and it resolves a 1.64 %/yr fee gap at monthly *t* = −5.2.

## What we tested

Realised annualised **tracking difference** of the two US spot-ETH lines — **ETHA**
(0.25 %/yr) and **ETHE** (2.50 %/yr) — against **ETH-USD** and against each other, over the
ETF era 2024-07-23 → 2026-06-30 (486 sessions), with block-bootstrap CIs, an era cut, a
calendar table, and a long/short capture trade carrying one execution lag, 10 bps one-way ×
NAV per leg and a swept short-leg borrow. The sponsor fees and the **3.0 %/yr net staking
yield are declared ASSUMPTIONS, not tape**, and are swept (0.12–0.25 % and 2.0–4.5 %/yr).
Every Sharpe is excess of BIL, and every resolution limit is published as a **ruler sweep**
(block 5 → 63 plus a non-overlapping monthly reading) rather than a single width.
**Coverage, stated up front:** the 2024-07-23 cohort has ~9 lines and we measure **2** —
FETH, ETHW, ETHV, QETH, EZET and CETH are not in cache — and the pair we do measure is the
cohort's cheapest against its dearest, i.e. its widest fee gap.
**Dedup:** distinct from **618-gbtc-premium-cycle** (the
*pre-conversion* closed-end discount, where this study's window begins), **913** (tracking-
difference *persistence* on conventional index funds), **959** (the *bitcoin* wrapper fee
race), **958** (futures-vs-spot basis), **378** (NAV premium mean-reversion — a nuisance
here, not the subject) and **209 / 582** (ETH *return* and gas-fee studies, not cost of
ownership).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why an ETF that owns coins still leaves money on the table, why the coin is the wrong yardstick, and what the two funds' own price gap reveals instead |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the tracking-difference estimator, the ±5-to-±18 %/yr ruler sweep behind the resolution limit, why the i.i.d. *t* is discarded and HAC is the flattering direction, the ledger and its sweeps, the borrow sweep, and the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`unstaked/`](unstaked/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
