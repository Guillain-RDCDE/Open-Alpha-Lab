# Study 132 — Yield-Curve-Steepener

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Q5−Q1 slope-quintile spread = **+2.37 bps/day**, HAC *t* = **+0.64** (all horizons sub-threshold); direction right, pattern non-monotone, evidence insufficient. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Timer Sharpe **0.250** vs buy-and-hold **0.252** across all cost levels; spread *t* = −0.27; only ~1.7 switches/year and still trails passive. |
| **Beats buy-and-hold?** | ![No](https://img.shields.io/badge/Beats_buy--and--hold%3F-No-8b949e?style=flat-square) | Active − passive spread = **−0.12 bps/day** (*t* = −0.27); the inverted-curve regime (13.5% of days) is too rare and heterogeneous to time reliably. |

> **In one sentence:** the Treasury yield-curve slope sends a directionally correct but statistically faint signal about next-day TLT returns — the Q5-Q1 spread is positive but far below the inference bar, and the binary timing overlay cannot consistently beat buy-and-hold over 23 years of daily data.

## What we tested

The bond-market version of a term-premium timing rule: when the 10Y Treasury yield is well above the 3-month T-bill rate (a *steep* curve), long bonds are fairly compensated for duration risk and should outperform; when the curve is flat or inverted, step aside. We take the simplest testable form — raw 10Y-minus-3M spread as the signal, TLT as the instrument — and pit it against an honest buy-and-hold TLT baseline over 2002-2026 (n = 6,000 daily observations, encompassing the 2006-07 flattening, 2008 crisis steepening, 2022-24 deep inversion, and re-steepening). Three tests: quintile-sorted forward TLT returns (Q5 vs Q1), a binary timing overlay, and regime-conditioned Sharpe statistics. A synthetic tape with tunable slope-signal serves as the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the curve history, the quintile chart, the timing overlay vs buy-and-hold in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stats, bootstrap Sharpe CI, regime breakdown, cost sweep, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`yield_curve_steepener/`](yield_curve_steepener/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
