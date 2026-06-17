# Study 238 -- Betting-Against-Beta

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

> Does leveraging low-beta and shorting high-beta (BAB) really print money?

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | BAB earns **+4.44%/yr**, HAC *t* = **+1.386** (|*t*| < 2). Frazzini-Pedersen (2014) provides strong prior on broad universes, but 178 months of large-cap survivors do not independently clear the bar. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Sharpe **+0.319**, max drawdown **-27.1%**, leverage avg **1.57x** (uncounted cost), brutal 2023 reversal (-25.9%). Leverage assumptions are stated but financing costs are not modelled. |
| **Survivorship-biased?** | ![Named](https://img.shields.io/badge/Named-8b949e?style=flat-square) | Universe = current S&P 500 projected backwards (96 tickers). Failed high-beta names -- the natural short -- are absent; results are **upper-bound estimates**. |

> **In one sentence:** the BAB premium is plausible in theory (leverage-constrained investors overcrowd high-beta stocks) and well-documented on broad universes, but on a 15-year large-cap survivor panel it shows up below the statistical bar -- Weak signal, Fragile tradability, leverage cost un-modelled.

## What we tested

Frazzini & Pedersen (2014): rank stocks monthly by trailing 252-day beta; go long the low-beta
half (leveraged to unit beta, cap 2.0x) and short the high-beta half (deleveraged to unit beta).
The portfolio is ex-ante beta-neutral. Panel: 96 large-cap S&P 500 names, yfinance daily prices
2010-2025 (178 monthly observations). Risk-free rate = 3%/yr constant. Universe is
survivorship-biased -- we name it and treat results as upper bounds.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the leverage-constraint mechanism in plain language, synthetic positive control, real panel BAB returns, honest verdict |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | rolling beta sweep, year-by-year breakdown, equity curve and drawdown, leverage sensitivity, survivorship discussion |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`betting_against_beta/`](betting_against_beta/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
