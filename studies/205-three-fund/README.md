# Study 205 — Three-Fund

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | Sharpe **0.737** (real, *t* = +5.17), but *lower* than both baselines: US-only (0.857) and 60/40 (0.929). Three-fund trails US-only by **−3.48%/yr** (HAC *t* = −4.90, Bonferroni-significant); vs 60/40: +1.31%/yr (*t* = +1.70, not significant). International attribution: VXUS sleeve subtracted **−2.01%/yr** (*t* = −3.25, significant). |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Annual rebalancing makes costs essentially irrelevant (5 bps one-way barely moves the Sharpe). But the allocation underperforms both baselines on a risk-adjusted basis. "Implementable" is not the same as "better." |
| **International helped?** | ![Hurt](https://img.shields.io/badge/Hurt-c0392b?style=flat-square) | The VXUS sleeve subtracted −2.01%/yr vs holding VTI instead (*t* = −3.25, Bonferroni-significant). 2011–2026 was an exceptional decade for US equity and a poor one for international. |

> **In one sentence:** the three-fund portfolio is sound theory and trivially implementable, but on the 2011–2026 VTI/VXUS/BND live tape it was the worst of three alternatives on Sharpe — driven by international equity underperforming US equity in every sub-period.

## What we tested

The **Bogleheads three-fund portfolio** (60% VTI / 30% VXUS / 10% BND, annual rebalance, 5 bps one-way cost) against two baselines: **100% US equity (SPY)** and **60/40 (SPY+BND)**. Data: daily adjusted-close prices from Yahoo Finance, 2011-02-01 to 2026-06-15 (3,865 trading days, ≈ 15 years). Inference: Newey-West HAC t-stats on annual return differences; Bonferroni correction across 3 simultaneous tests (threshold |*t*| ≥ 2.394); circular block-bootstrap Sharpe CI. International attribution: three-fund vs 90% VTI / 10% BND (no VXUS sleeve). Synthetic positive control: planted international alpha confirms the engine detects the benefit when it exists.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | equity curves, drawdown comparison, VXUS attribution in plain language, tradability |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC pairwise t-tests, Bonferroni correction, bootstrap Sharpe CIs, sub-period breakdown, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`three_fund/`](three_fund/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
