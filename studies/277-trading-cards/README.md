# Study 277 — Trading-Cards

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**Are Pokemon/sports cards an investable asset class?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Card CAGR **5.2%** < S&P **9.0%**, Sharpe **0.35** < **0.62**, deeper drawdown (**−44%**); structural alpha HAC t = **1.33** (< 2), bootstrap CI includes zero. Index is survivorship-biased upward. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Net of ~15% one-way auction + grading frictions the CAGR is **−0.9%/yr**; no liquid, low-cost vehicle, and the low correlation bought no hedge in 2022. |
| **Busted?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The entire outperformance is the 2020–2021 mania, which fully round-tripped; ex-boom the index compounds at **2.1%/yr**. |

> **In one sentence:** trading cards looked like an asset class for exactly two pandemic years — strip the 2020–2021 boom, pay the brutal auction/grading frictions, and the "asset" loses money while underperforming a plain index fund on every risk-adjusted axis.

## What we tested

We pair a curated annual **graded-card price index** (hardcoded in `data.py`, base
100 = 1990, capturing the documented long grind → 2020–2021 mania → 2022–2023 crash
→ partial recovery) with the S&P 500's calendar-year **price** return (cached ^GSPC),
1991–2025. We compute return/risk (CAGR, vol, Sharpe, drawdown), the structural
**alpha over equity beta** with **Newey-West (HAC)** standard errors, net out
realistic collectible frictions, put a **block-bootstrap** confidence interval on the
alpha, and show the whole result is one boom by recomputing ex-2020-2021. The
synthetic positive control confirms the engine finds a card premium when one is
planted; the real tape confirms there is none here after costs.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the boom-and-crash chart, cards vs stocks, the friction reckoning, the failed 2022 hedge — in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | Newey-West alpha, net-of-cost alpha, block bootstrap, sub-period CAGR, the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`trading_cards/`](trading_cards/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
