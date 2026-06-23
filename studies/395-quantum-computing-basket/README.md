# Study 395 — Quantum-Computing-Basket ⚛️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the quantum basket genuinely out-earn, risk-adjusted? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The *raw* spread is enormous — the four pure-plays beat SPY by **+104.6%/yr** — but at HAC **t = 1.23** (n = 61; 1.19 vs QQQ, 1.12 vs the diversified ETF) it **fails t ≥ 2**, and **no single name clears t = 2** either. Worse, the basket's **Sharpe (0.75) is *below* the S&P 500's (0.91)**: the big number is bought entirely with 160%-vol risk. Survivorship-tilted (de-SPAC survivors), one short regime. A positive-but-insignificant, risk-unrewarded point estimate, not an edge. |
| **Tradability** — could you actually hold it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Costs are trivial (net Sharpe 0.74 vs 0.75). **Risk** is the killer: a **160% annualised vol** and an **−84% drawdown** (single names −86% to −96%) make the basket un-allocatable at NAV scale — and risk-adjusted it *loses* to both SPY and the diversified **QTUM ETF (Sharpe 1.06)** you could have bought from day one. The +59%/yr is a number you couldn't have held through the −84% draw. |
| **The next big thing?** | ![Hype-cycle](https://img.shields.io/badge/Next_big_thing%3F-Hype--cycle-8b949e?style=flat-square) | A real technology, no profits yet, and a stock cohort that delivers a giant CAGR, a giant drawdown, a **sub-market Sharpe**, and a *t* that can't reach significance — a **high-variance lottery dressed as a thesis**. A synthetic null reproduces the spread from pure beta-plus-noise (t = 0.18). [Study 393 (AI-Datacenter)](../393-ai-datacenter-basket/) in a pre-revenue costume. |

> **In one sentence:** the quantum-computing basket really did return **+59%/yr** vs the market's **+14%** — but it did so at **160% volatility** with an **−84% drawdown**, a **Sharpe (0.75) *below* the S&P 500's**, and a HAC *t* of **1.23** that can't separate the spread from zero; strip the hype and you're left with a survivorship-tilted lottery over one short regime that a synthetic null reproduces from pure beta-plus-noise (t = 0.18), while the diversified **QTUM ETF (Sharpe 1.06)** quietly does the theme better — real raw return, no certifiable edge, undeployable as a holding.

## What we tested

The viral **"ride the quantum revolution"** trade: hold the four **pure-play** quantum-computing
stocks — IONQ (trapped-ion), RGTI (superconducting), QBTS (annealing), QUBT (photonic) —
equal-weight, monthly-rebalanced, and ride the "next big thing" instead of the boring index. Over
**61 months (2021-05 → 2026-05)** of yfinance monthly total returns we race the four against
**SPY** (the market), **QQQ** (the tech tape), and the **diversified QTUM ETF** (the sane way to
"play quantum"). The headline CAGR is spectacular, so the question is *what it costs*: we judge it on
**Sharpe, volatility and drawdown** (the hype-cycle signature) and on the **HAC *t*-stat of the
spread** (is the giant number even significant?), then decompose it name-by-name. A deterministic
synthetic null — a fat-tailed hype basket with **no** risk-adjusted edge — shows that pure
beta-plus-noise reproduces a large spread with an insignificant *t*. Survivorship (the de-SPAC
cohort that *survived* to a listing) is named on the **Signal** axis. (Same thematic-basket family as
[Study 393](../393-ai-datacenter-basket/) and [Study 334](../334-ark-innovation/); here the critique
is risk & sample size rather than ex-post name selection.)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a +59%/yr basket can still be a *worse* deal than the index, what an −84% drawdown does to you, and why "next big thing" ≠ "buy the basket" — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the Sharpe/vol/drawdown race, the HAC *t* of the spread vs SPY/QQQ/QTUM, the single-name decomposition, the survivorship caveat, and a synthetic null that reproduces the spread from zero risk-adjusted edge |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantum_computing_basket/`](quantum_computing_basket/). The four pure-plays are an explicit **surviving-cohort** selection (de-SPAC survivorship runs bullish), named on the Signal axis. **Not investment advice** — research & education; concentrated pre-revenue thematic risk is extreme (single names drew down −86% to −96%). See [LICENSE](../../LICENSE).*
