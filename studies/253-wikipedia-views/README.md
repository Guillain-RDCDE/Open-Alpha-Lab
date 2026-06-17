# Study 253 -- Wiki-Views

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Drought-minus-surge quintile spread HAC *t* = **-0.31** across 118 months; block-bootstrap Sharpe 95% CI = [-0.73, +0.54] (63% negative); no sub-period clears \|t\| = 1; random-portfolio null centred at zero. The Da-Engelberg-Gao attention prior has literature support, but our monthly mega-cap tape shows **no** forward information -- even the sign is wrong. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Gross spread already negative; ~89% monthly two-leg turnover; -45.8 bps/mo net at 10 bps one-way *before* short-leg borrow. There is no edge to harvest. |
| **Survivorship bias** | ![Named](https://img.shields.io/badge/Survivorship--biased-8b949e?style=flat-square) | Basket = current mega-cap survivors projected backwards; results are upper bounds, and the upper bound is already zero. |

> **In one sentence:** ranking a mega-cap basket each month by the surge in its Wikipedia page-views and betting on the published attention-reversal prior produces a spread that is statistically indistinguishable from zero (*t* = -0.31), with the wrong sign and ~89% turnover -- a folklore signal at the wrong frequency, in the wrong universe.

## The claim

> *Do surging Wikipedia page-views on tickers predict the next move?*

## What we tested

The Moat et al. (2013) / Da-Engelberg-Gao (2011) attention story: each month-end,
measure each name's **attention surge** -- the month-over-month log change in its
Wikipedia page-views -- and exploit the reversal prior by going long the bottom
("attention drought") quintile and short the top ("attention surge") quintile,
holding one month. The page-view side is a **curated anchor table** expanded into
a deterministic monthly panel (the Wikimedia API only starts 2015-07 and is
spike-dominated; a brittle live scrape would break the offline guarantee). The
returns side is a fixed mega-cap basket (yfinance monthly, total-return; offline
runs use a views-independent proxy -- the honest null). We pin the result against
(a) the equal-weight basket, (b) a random-portfolio control of identical leg
size, and (c) a 2017-2026 sub-period breakdown, plus a turnover cost sweep. A
deterministic synthetic positive control confirms the engine recovers a planted
attention-reversal premium (*t* = +17) when it exists.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the attention story in plain language, why "trade the views surge" is folklore at the monthly mega-cap frequency, and the drought-vs-surge race that goes nowhere |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC *t*-stats, bootstrap Sharpe CI, sub-period nulls, turnover cost sweep, random-portfolio control, the synthetic positive control, and the data-honesty caveat on curated page-views |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`wikipedia_views/`](wikipedia_views/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
