# Study 975 — Shrink the Matrix 🗜

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the sample covariance measurably bad out of sample? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | On the wide cross-section (40 names, 820 covariance parameters, 252 rows of history) the sample matrix promises 9.81% annualised volatility and delivers **13.07%** — it is **25%** optimistic, and its median condition number is **335** against 162 for the shrunk version. On the eleven-sector sleeve, where rows outnumber parameters 4 to one, the same optimism is only 11% — the problem is arithmetic, not equities. |
| **Tradability** — does shrinkage buy enough to be worth the code? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | **Ledoit-Wolf -> identity** cut realised volatility from 13.07% to **12.49%** — a **4%** saving, paired *t* = **+6.51** across 61 rebalances, winning 16% of them. It also turns over 1.14 against 1.54 per rebalance and holds a maximum weight of 24% against 32%. Three lines of algebra, no tuning parameter, no look-ahead. |

> **In one sentence:** The sample covariance matrix is not wrong so much as overconfident: on forty names estimated from a year of data it under-promises risk by **25%** and builds portfolios with 32% single-name weights, while a shrinkage estimator with no free parameters cuts realised volatility by **4%** — and on eleven sectors, where the arithmetic is comfortable, it changes almost nothing.

## What we tested

A covariance matrix estimated from fewer observations than it has parameters is not
merely noisy — it is *confidently* wrong, with eigenvalues spread far wider than the truth's,
and a minimum-variance optimiser is drawn to exactly the smallest of them because they look
like free risk reduction. **Ledoit and Wolf** proposed the fix in 2003-2004: shrink the sample
matrix toward a structured target (a scaled identity, or the constant-correlation matrix), with
an intensity computed analytically rather than tuned. This study tests it where it should
matter and where it should not: **eleven sector ETFs** (66 parameters, comfortable) and **forty
single names** (820 parameters, one year of history — 0.3 rows per parameter). Four estimators
— sample, both Ledoit-Wolf targets and the variances-only diagonal — are re-estimated on a
rolling window every quarter, and each builds the minimum-variance portfolio that is then held
out of sample; the scoreboard is realised volatility, with the promised-versus-delivered gap,
condition number, turnover and concentration alongside. Differences are tested **paired**,
because every estimator sees the same window and the same holding period.

The whole apparatus is validated against a factor panel whose true covariance matrix is known
in closed form, at four sample sizes, so the ordering on real data can be checked against the
ordering where truth is observable.
**Dedup:** distinct from **967-rolling-vs-expanding** (how much history, not which estimator),
**976-hierarchical-risk-parity** and **977-max-diversification** (different *allocation* rules
given a matrix), **171-naive-1-over-n** (skipping estimation altogether) and
**902-multi-factor-composite** (a factor model of returns rather than of risk).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a matrix can be unbiased and useless at the same time, the optimiser that promises quiet and delivers loud, and the three-line fix |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | both Ledoit-Wolf targets with analytic intensities, eigenvalue spectra and condition numbers, rolling out-of-sample minimum variance with paired tests, the long-only comparison and a known-truth simulation |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`shrinkage/`](shrinkage/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
