# Study 908 — Optimized-Roll Commodities 🛢️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the optimized-roll wrapper beat front-month on excess-of-cash Sharpe? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | USCI (optimized) wins the full-sample race against **every** benchmark — excess-of-cash Sharpe **+0.245** vs DBC +0.14 / GSG +0.05 / DJP +0.04, the *right sign* for a structural roll-yield edge, and PDBC corroborates the direction. But no return-difference clears **HAC *t* ≥ 2** (best +1.67 vs DJP), **every** paired-bootstrap Sharpe-advantage CI includes zero, and it is **not era-robust**: USCI is significantly *worse* through 2016-2020 (−7.22 %/yr vs DBC at *t* = **−3.04**; −4.79 %/yr vs DJP at *t* = −2.13) before recovering post-2021. A directionally-right, sensible pickup a 16-year two-regime tape can't establish as real. *Short-history / two-regime caveat named on this axis.* |
| **Tradability** — can you bank the roll edge? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Costs don't kill it (the race is already net of expense ratios — USCI's steep 1.03 % vs GSG 0.48 % — and both legs are buy-and-hold), and USCI even ended ahead with the shallowest drawdown (−64 % vs GSG −78 %). But there's no *bankable* edge: the isolation advantage's CI straddles zero, the sign flips by regime, and much of the raw Sharpe gap is a **hidden composition tilt** (USCI's lighter energy weight → 14.7 % vol vs GSG's 21.3 %), not a robustly paid roll premium. "Buy the optimized wrapper to pocket the roll" is a contango-regime bet wearing a free-lunch label. |

> **In one sentence:** the optimized-roll wrapper (USCI) really did dodge the front-month
> "contango tax" and end 16 years ahead on excess-of-cash Sharpe (+0.245 vs +0.04-0.14) with a
> shallower drawdown — but the edge is **not statistically robust and not era-robust** (it lost
> significantly through 2016-2020, *t* = −3.0), so the roll-yield "free lunch" is really a
> **contango-regime bet**: a real-looking direction the tape can't promote past **Weak / Mirage**.

## What we tested

A front-month commodity index rolls naively up a rising futures curve and bleeds a negative
**roll yield** (the "contango tax"); an **optimized-roll** index (USCI — the SummerHaven Dynamic
index: hold the 14 most-backwardated of 27 commodities, each in its cheapest-carry contract)
dodges much of it. We race USCI against **DBC** (DB "Optimum Yield", semi-optimized — the
primary benchmark), **GSG** (S&P GSCI, naive front-month) and **DJP** (Bloomberg, front-month),
with **PDBC** (optimized, 2014-11+) as a corroborating cousin, on yfinance **total-return**
closes, **2010-09 → 2026-06** (190 months; USCI's 2010-08 inception gates the start). Every leg
is taken **excess of cash (BIL)** so the shared collateral T-bill yield (~5 %/yr in 2023-26) is
netted out of both sides and only the spot + roll difference remains. Inference is a paired
**circular-block bootstrap** 95 % CI on the excess-of-cash **Sharpe advantage** plus a
**Newey-West HAC *t*** on the monthly return difference, with a **three-era cut** (deep-contango
2010-2015 · recovery 2016-2020 · backwardation 2021-2026) as the decisive robustness check and a
costed layer (total returns already net expense ratios; an incremental bid-ask charge is added).
A deterministic synthetic world with a **planted, tunable roll edge** proves the estimator
recovers a genuine constant edge (+3 %/yr → *t* = +4.27) and stays dark at the null.
**Dedup:** [35-contango](../35-contango/) *times* the curve (in/out of the market); this study
times nothing, it races two always-invested wrappers. [794-commodity-carry](../794-commodity-carry/)
is a **cross-sectional** long-short of single commodities on carry; here we buy whole **packaged
indices**. [661-uso-roll-decay](../661-uso-roll-decay/) is the roll decay of one front-month
vehicle (USO); this is a **broad multi-commodity** race. [226-crude-seasonality](../226-crude-seasonality/)
is a single-commodity calendar effect — unrelated. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the "contango tax" is, how an optimized index dodges it, and why USCI's 16-year win is really a regime bet in disguise |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the excess-of-cash Sharpe race, the paired block-bootstrap advantage CIs, the HAC *t* on the return difference, the era sign-flip, the cost and composition-tilt decomposition, and the planted-edge synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`opt_roll/`](opt_roll/). Every leg is measured excess of cash (BIL); the Sharpe
advantage is bootstrapped pairwise and the return difference carries a Newey-West HAC *t*.
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
