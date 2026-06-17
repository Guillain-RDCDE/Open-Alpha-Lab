# Study 241 — Buy-the-Dip

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

> Is "buy the dip" actually a rule that beats just staying invested?

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Every threshold underperforms buy-and-hold; HAC *t* on daily excess returns ranges from −1.88 (2% dip) to **−4.05** (10% dip). The underperformance is statistically real and in the wrong direction. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The rule is tradable in principle (SPY, low friction) but there is no edge to trade. Cash drag destroys −1.8% to −7.7%/yr of CAGR vs buy-and-hold depending on threshold. Even a 4% cash rate does not close the gap. |
| **Myth check** | ![Busted](https://img.shields.io/badge/BUSTED-8b949e?style=flat-square) | Buying the dip systematically underperforms staying invested. Time in the market beats timing the market, across every threshold and every sensitivity tested. |

> **In one sentence:** waiting for dips to buy SPY costs you −1.8% to −7.7%/yr of CAGR versus always-invested buy-and-hold — statistically confirmed underperformance, not a timing edge.

## What we tested

Systematic dip-buying on SPY (1993–2026, 33.4 years): hold cash until the running
drawdown from the all-time high reaches a threshold (grid: 2%, 5%, 10%, 15%, 20%),
buy at the next day's open, exit at the next open after a new ATH close. One-way cost
= 1 bp. Benchmarks: always-invested buy-and-hold (+10.86%/yr, Sharpe 0.648) and
monthly DCA. Inference: HAC *t*-stat on daily excess returns (dip-buyer minus BH).
Sensitivity: cash earning 4%/yr (T-bill proxy). Null confirmed on a synthetic tape
with planted mean-reversion (the rule does better when dips genuinely recover faster
than random — but even then, cash drag remains the dominant force in long bull markets).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the folk rule feels right, why cash drag kills it, the real equity curves, the intuition |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | threshold grid, HAC t-stats, DCA comparison, cash-rate sensitivity, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`buy_the_dip/`](buy_the_dip/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
