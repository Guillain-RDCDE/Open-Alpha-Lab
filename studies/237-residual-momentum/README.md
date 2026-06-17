# Study 237 -- Residual-Momentum

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

> *Does momentum on market-adjusted (residual) returns dodge the crashes?*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Residual momentum spread HAC *t* = **+1.20** over 365 months (below the bar of 2.0); winner excess over market *t* = +1.96. Winner alpha after beta adjustment = +3.04%/yr (positive direction). Effect turned negative in 2016-2026 (-0.9%/yr). Survivorship-biased upper bound. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | ~23.5% monthly turnover from residual rebalancing; 10 bps one-way erodes net spread to *t* = +1.05; loser portfolio beta = 1.24 amplifies crash losses; universe limited to ~70 large-cap names. |
| **Survivorship bias** | ![Named](https://img.shields.io/badge/Survivorship--biased-8b949e?style=flat-square) | Universe = current large-cap basket backwards; delistings absent -- all results are upper bounds. |

> **In one sentence:** stripping market beta from the momentum signal produces a spread in the right direction (+4.3%/yr gross) and does reduce systematic crash exposure (spread beta = -0.17 vs market), but the t-stat of +1.20 never clears the inference bar, the effect has vanished in the most recent decade, and high turnover eats what little gross edge exists -- a theoretically elegant refinement that does not survive rigorous scrutiny on the modern survivorship-biased tape.

## What we tested

The Blitz, Huij & Martens (2011) recipe: each month, estimate each stock's CAPM beta via a rolling 36-month OLS on SPY. Compute the trailing 11-month cumulative *residual* return (actual minus beta × SPY, skip 1 month). Sort the large-cap universe by residual signal; long the top quintile (residual winners), short the bottom quintile (residual losers); hold 1 month and rebalance. We compare directly against raw 12-1 momentum on the same universe, run a beta decomposition, test crash-month behaviour (2009-03, 2020-03), analyse sub-period decay, and quantify turnover cost drag. A deterministic synthetic positive control confirms the engine recovers planted residual-momentum premium.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the intuition behind beta-stripping, the winner-loser race in plain language, why the crash shield is partial, and why the edge has faded |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC *t*-stats, beta decomposition, residual vs raw comparison, sub-period breakdown, turnover cost sweep, crash-month analysis, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Sibling studies: [Study 24 -- Stampede](../24-stampede/) (raw momentum), [Study 196 -- Long-Term-Reversal](../196-long-term-reversal/) (long-horizon reversal), [Study 33 -- Slingshot](../33-slingshot/) (short-term reversal).*

*Engine: [`quantlab/`](../../quantlab/) + [`residual_momentum/`](residual_momentum/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
