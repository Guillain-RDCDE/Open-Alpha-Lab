# Study 513 -- Size-Effect

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

> Do small-cap stocks really out-earn large-caps -- the original Banz (1981) anomaly?

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Small-minus-large earns **-1.08%/yr**, HAC *t* = **-0.48** (wrong sign), placebo *p* = **0.61**. On a 40-name survivor basket over the large-cap decade 2001-2026 there is no size premium. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Sharpe **-0.09** gross, **-0.17** net; short borrow alone drags the book to **-2.08%/yr**. Nothing to trade. |
| **January concentration?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The anomaly's signature January small-cap pop is **absent and reversed** here: Januaries were the *worst* month for the spread (-1.0%/mo, Welch *p* = 0.10). |

> **In one sentence:** the original size anomaly -- small beats big -- vanishes (and inverts) on a 40-name survivor basket over the mega-cap-led 2001-2026 era: the long-short loses -1.1%/yr at HAC *t* = -0.48, the placebo can't tell it from noise, the famous January pop is reversed, and costs only deepen the hole -- None signal, Mirage tradability.

## What we tested

Banz (1981) / Fama-French SMB: each month-end rank the basket by market capitalisation, go
long the small half and short the large half (equal-weight), with one execution lag (signal
through month *m*, return realised in month *m+1*). Dollar-neutral **and** beta-neutral books;
costs = 10 bps x turnover + 100 bps/yr short borrow. Basket: 40 names spanning mega-caps to
genuine small/mid-caps, yfinance daily prices 2001-2026 (304 monthly observations). A
label-shuffle placebo, a January slice, an early-vs-late decay slice, and a 25-seed synthetic
positive control that proves the engine recovers a planted premium. The basket is
**survivorship-biased** -- we name it and treat results as upper bounds.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the small-vs-big idea in plain language, the synthetic positive control, the real-basket long-short, the busted January effect, the honest verdict |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | cap-ranking mechanics, dollar- vs beta-neutral spreads, label-shuffle placebo, early-vs-late decay, equity curve & drawdown, survivorship discussion |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`size_effect/`](size_effect/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
