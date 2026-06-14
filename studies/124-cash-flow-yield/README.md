# Study 124 — Cash-Flow-Yield

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | OCF-yield hedge (Q5−Q1) **−0.6%/yr**, HAC *t* = **−0.22**; EY hedge −1.6%/yr, *t* = −0.77. Neither clears the \|*t*\| ≥ 2 bar on 18 years. Hit rate 44% for OCF — below a coin. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No statistically significant spread at any reasonable cost assumption. Annual rebalancing is cheap but there is no edge to execute. |
| **OCF beats EY?** | ![No](https://img.shields.io/badge/OCF_beats_EY%3F-No-8b949e?style=flat-square) | Head-to-head IC comparison: OCF IC mean +0.007 (*t* = +0.21) vs EY IC mean +0.018 (*t* = +0.69); difference *t* = −0.61. No statistically significant advantage for cash-flow yield over the simpler earnings yield. |

> **In one sentence:** on 18 years of survivorship-biased S&P 500 data, sorting by operating cash-flow yield produces a statistically invisible hedge (-0.6%/yr, t = -0.22) that does not outperform the simpler earnings-yield sort, consistent with post-publication decay of fundamental-factor anomalies in large-cap equities.

## What we tested

The "OCF is better than earnings" thesis: because reported earnings can be massaged via
accruals, operating cash flow (money actually collected) is a harder signal to fake.
Sorting stocks by OCF/market-cap (the free-cash-flow equivalent of P/E's inverse) should
identify genuine value firms that outperform. The head-to-head against earnings yield
(NI/market-cap) tests the incremental claim — that this purification beyond P/E adds alpha.
We implement a standard annual quintile sort on current S&P 500 members (EDGAR 10-K data,
2007–2024), with a one-year reporting lag to avoid look-ahead. The universe is survivorship-
biased (current membership projected back), so findings are upper-bound estimates. A
deterministic synthetic panel with tunable premiums provides the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the "cash is king" thesis in plain language, why OCF ought to beat EY, what the quintile sort actually shows, the survivorship caveat |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stats per year, Spearman IC breakdown, OCF vs EY head-to-head, the synthetic positive control confirming the engine works |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`cash_flow_yield/`](cash_flow_yield/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
