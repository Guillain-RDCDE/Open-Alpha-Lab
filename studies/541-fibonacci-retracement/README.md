# Study 541 — Fibonacci-Retracement 🌀

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do pullbacks reverse *at* the Fib levels? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Pooled across **225** Fibonacci-depth swings on 8 index/ETF tapes over 33 years, the "trend resumes at the Fib level" bet earns **−15 bps** (one-sample *t* **−0.27**, HAC −0.23) — indistinguishable from zero and from a coin at the same swings (coin *p* 0.61). And it is **no better than a placebo** of arbitrary interleaved fractions: the Fib-minus-placebo edge is *t* **0.80**, and it **flips sign** across the grid (+1.53 to −1.00). No *t* ≥ 2 anywhere. |
| **Tradability** — does the level pay? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Nothing to harvest: the reversal bet is flat-to-negative before costs (**−15 bps** gross → **−17 bps** net at 1 bp/side), the per-tape edge is pure noise (**−478 to +768 bps**), and the sign is unstable across ZigZag threshold and horizon. |
| **"Fib levels are special"?** | ![Not_Supported](https://img.shields.io/badge/Fib_levels_special%3F-Not_Supported-8b949e?style=flat-square) | Arbitrary non-Fibonacci fractions (0.31 / 0.44 / 0.56) in the *same* depth band do just as well. A synthetic control proves the engine **would** catch a planted level-effect (edge *t* up to **+13.8** at 25 seeds), so this is a true negative, not a broken test. |

> **In one sentence:** the famous 38.2% / 61.8% Fibonacci retracement levels do **not** mark where a pullback stops — across 225 Fib-depth swings on 8 broad tapes over 33 years the "reversal at the level" bet earns nothing (*t* −0.27), a coin at the same swings matches it, and *arbitrary* interleaved fractions do just as well (edge *t* 0.80, sign-unstable) — even though the same engine lights up at *t* +13.8 when we *plant* a level-effect.

## What we tested

Chartists draw **Fibonacci retracements** from a swing high to a swing low and claim price
reverses **at** the 23.6% / 38.2% / 50% / 61.8% levels of the move. We mark swings with a standard
**5% ZigZag**, compute each pullback's *realised* retracement of its impulse leg (look-ahead-free
at the confirmation bar), and bet the trend **resumes** at that pullback — entering **one bar
after** the pivot is confirmed and holding **20 days**. The decisive test is a head-to-head: the
forward return when a pullback lands **near a Fibonacci depth** versus a **placebo** of arbitrary
non-Fibonacci fractions (0.31 / 0.44 / 0.56) *interleaved in the same depth band*, so the only
difference is the exact fractions. We report the pooled one-sample and HAC *t*, a same-bars
**coin placebo**, the Fib-minus-placebo two-sample edge, a ZigZag-threshold × horizon × tolerance
robustness sweep, gross and net (1 bp/side), and **two seed-robust synthetic controls** — a planted
level-effect the engine must catch, and a genuine price-path null it must stay flat on. *Distinct
from [445 Elliott Wave](../../445-elliott-wave/) (tests the wave-3 count, not the bare level) and
from [93 Round Numbers](../../93-round-numbers/) (round price levels, not swing-retracement
fractions).*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a Fibonacci retracement is, why 38.2% / 61.8% "should" matter, and why on real charts a bounce at a Fib level is no more likely than at any round-ish level — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the ZigZag swings, the realised-retracement binning, the Fib-vs-placebo two-sample edge, the same-bars coin placebo, the threshold × horizon robustness grid, costs, and the two seed-robust synthetic controls |

The fingerprinted real-data run (8 tapes, combined fp `11844738632b`, as-of 2026-06-30) is in
[docs/results.md](docs/results.md); the offline machinery proof runs on the deterministic synthetic
worlds in [`fibonacci_retracement/data.py`](fibonacci_retracement/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`fibonacci_retracement/`](fibonacci_retracement/). Tapes are auto-adjusted (total-return) daily closes; broad price-index/ETF survivors, no cross-sectional single-name sort. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
