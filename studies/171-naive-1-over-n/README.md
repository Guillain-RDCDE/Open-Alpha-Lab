# Study 171 — Naive-1-Over-N

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does optimisation beat 1/N out of sample? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | 1/N leads **every** optimiser on Sharpe (1.179 vs 1.084–1.111) across all lookbacks; max-Sharpe ekes +0.96%/yr raw CAGR. n=4 annual OOS obs — too small to call definitively. |
| **Tradability** — is there a harvest worth implementing? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | 1/N is the cheapest strategy; optimisers face turnover costs and unstable weights. No robust implementation edge over just holding equal weight. |
| **Optimisation is the mirage?** | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | On Sharpe: DeMiguel et al.'s headline holds. On raw CAGR in this window: max-Sharpe wins by a sliver. The honest answer is: *we can't tell on 4 years of data.* |

> **In one sentence:** the 1/N equal-weight portfolio beats all three Markowitz optimisers on risk-adjusted return (Sharpe) across every lookback and cost scenario, replicating DeMiguel, Garlappi & Uppal (2009) directionally — but the out-of-sample window is far too short for a verdict that survives a t-test.

## What we tested

The 2009 *Review of Financial Studies* landmark: DeMiguel, Garlappi & Uppal found that *"the out-of-sample performance of the 1/N portfolio strategy is difficult to beat consistently"* — because estimation error on the expected-return vector swamps the optimiser's theoretical advantage. We replicate the tournament on the eleven SPDR sector ETFs (XLB through XLY, 2018–2026), comparing 1/N equal-weight against **min-variance**, **max-Sharpe (tangency)**, and **mean-variance (λ=3)** — all estimated from rolling 60-month windows — with realistic rebalance costs and HAC inference on the annual return differential.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the estimation-error story in plain language, why 1/N is so hard to beat, the cost penalty on optimisers |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | rolling-window out-of-sample engine, per-lookback / per-cost Sharpe table, HAC *t* on pairwise diffs, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`naive_1_over_n/`](naive_1_over_n/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
