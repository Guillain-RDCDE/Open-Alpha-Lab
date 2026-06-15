# Study 196 -- Long-Term-Reversal

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Loser-winner spread HAC *t* = **+0.95** (not significant); the loser's +5.5%/yr gross excess is entirely beta-driven (beta = 1.31 vs winner = 1.02); alpha = -0.10%/yr. Effect decayed from *t* = +2.33 pre-2005 to *t* = +0.76 post-2015. Survivorship-biased upper bound. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Monthly rebalance at ~14% turnover; 10 bps one-way erodes a gross spread already below the inference bar; loser portfolio is effectively a leveraged-beta position, not a diversifying alpha. |
| **Survivorship bias** | ![Named](https://img.shields.io/badge/Survivorship--biased-8b949e?style=flat-square) | Universe = current S&P 500 backwards; true losers (delistings) are missing -- all results are upper bounds. |

> **In one sentence:** the De Bondt-Thaler long-horizon reversal shows a positive gross spread on the survivorship-biased S&P 500 tape but the spread *t*-stat never clears 1, the excess return is explained by the loser portfolio's higher market beta, and the effect has decayed sharply since its 1985 publication -- a real-but-shrinking, beta-laden, capacity-thin effect.

## What we tested

The De Bondt & Thaler (1985, *"Does the Stock Market Overreact?"*) recipe: each month, sort the S&P 500 universe by trailing 36-month total return; long the bottom quintile (past *losers*), short (or compare against) the top quintile (past *winners*); hold for 1 month and repeat. We pin the loser portfolio against (a) the equal-weight market, (b) a random-portfolio control of identical size, and (c) a beta decomposition to see how much of the raw outperformance survives a market-risk adjustment. We also test the 60-month formation window, report a sub-period breakdown from 1993 to 2026, and quantify the monthly turnover cost drag. A deterministic synthetic positive control confirms the engine recovers planted reversal when it exists.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the overreaction story, the loser-winner race in plain language, the beta-dressed alpha trap, and why the edge shrank after 1985 |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC *t*-stats, bootstrap Sharpe CI, beta decomposition, sub-period decay, turnover cost sweep, survivorship-bias caveat, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`long_term_reversal/`](long_term_reversal/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
