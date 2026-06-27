# Study 530 -- Book-To-Market-Value

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

> Do cheap (high book-to-market) stocks really beat expensive (low book-to-market) ones -- the canonical Fama-French value premium?

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The long-value / short-growth hedge earns **-5.62%/yr** (the *wrong* sign), HAC *t* = **-0.554**, label-shuffle placebo *p* = **0.583**. The canonical value premium is absent -- even slightly negative -- on this tape. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Net of costs **-6.15%/yr**, Sharpe **-0.25**, built on **3 annual observations** (yfinance serves only ~4-5 balance sheets/name). Nothing to trade; HAC inference is low-powered by construction. |
| **Did canonical value survive the modern survivor tape?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Over 2022-2024 signal years (the AI / mega-cap-growth era), cheap names *trailed* growth. A faithful synthetic control (seed-robust *t* > 7) proves the engine works -- the market simply did not pay value here. |

> **In one sentence:** the most famous factor in finance -- Fama-French book-to-market value -- replicated honestly on a 40-name large-cap survivor basket over the only window yfinance fundamentals allow (2022-2025) produces a *negative*, statistically-zero premium, a textbook None x Mirage with the value-is-dead regime Busting the third axis.

## What we tested

Fama & French (1992): rank stocks annually by **book-to-market** (book equity / market cap),
go long the high-B/M quintile (cheap value, Q5) and short the low-B/M quintile (expensive
growth, Q1). Book equity from yfinance `Stockholders Equity`, shares from `Ordinary Shares
Number`, market cap = shares × fiscal-year-end price. One execution lag: the position is held
for the full calendar year that begins *after* the 10-K is public. Costs: 10 bps one-way
× turnover + 50 bps/yr borrow on the short leg. Inference: Newey-West HAC *t* plus a 2000-draw
label-shuffle placebo. Basket: 40 large-cap S&P 500 survivors -- named and treated as an upper
bound. The binding constraint is the **3-year** fundamental history yfinance exposes.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what book-to-market is in plain language, why value "should" win, the synthetic control, the real-tape result, the honest verdict |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the quintile sort, HAC inference, label-shuffle placebo, costs × turnover, synthetic-control sweep + seed robustness, the data-limitation discussion |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`book_to_market_value/`](book_to_market_value/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
