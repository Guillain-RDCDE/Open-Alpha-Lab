# Study 176 — Hot-Hand 🎲

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | Up-streak continuation (hot-hand): **NONE** — HAC *t* = **+0.15**, coin. Down-streak reversal (gambler's fallacy right!): **REAL** — HAC *t* = **+4.91** on n = 742 trades, robust to 2008–2009 exclusion (*t* = +5.24). |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Down-reversal survives costs (net *t* = +4.69 at 1 bp) but delivers only ~4.6%/yr CAGR vs S&P 500 B&H ~8.5%. It fires ~22×/yr, holding cash otherwise — underperforms simple buy-and-hold. |
| **Streak length info?** | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Longer up-streaks carry zero incremental signal; longer down-streaks show stronger reversal but vanishing n (74 events at length 5, 23 at length 6) — the Bonferroni-corrected binomial confirms 5 cells survive, all in the down direction. |

> **In one sentence:** after N consecutive up-days the market is a coin — neither the hot-hand nor the gambler's fallacy wins — but after N consecutive down-days the gambler is right: the bounce is real, *t* = 4.91, though too slow and sparse to beat simply staying invested.

## What we tested

Two famous folk beliefs pit themselves against the same daily S&P 500 tape (1993–2026, n = 8,419 days): the **hot-hand** says "it's on a roll — keep going" (follow the streak); the **gambler's fallacy** says "it's due for a turn — fade it" (reverse the streak). We test both for streaks of length 1–6 in both directions, using next-day log returns vs an unconditional buy-and-hold baseline, with exact binomial tests for continuation probability and HAC t-stats on the excess return. A Bonferroni correction across 12 simultaneous sub-tests separates signal from noise. A synthetic tape with tunable daily AR(1) autocorrelation serves as the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the two folk claims, why the hot-hand is a coin, why the gambler gets lucky on down-streaks, and why neither pays better than holding |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-cell HAC *t*, binomial tests, Bonferroni correction, cost sweep, synthetic positive control, regime robustness |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`hot_hand/`](hot_hand/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
