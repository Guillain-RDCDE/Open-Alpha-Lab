# Study 265 -- IPO-Volume

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Full-sample predictive regression of next-year S&P 500 price return on the standardized IPO-volume "froth" z-score gives slope = -0.0019, **HAC *t* = -0.07** across 41 years -- a flat zero. The +0.32 *contemporaneous* correlation is a coincident-not-leading artifact; the negative recent sub-period slopes (*t* = -2.1, -6.2) are *n* ~ 12 subsample mirages. A synthetic positive control recovers a planted slope at *t* = -4.9, so the null is real, not a broken engine. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The tradable long/flat wrapper (long the index next year when froth < 0, flat otherwise) returns +8.3%/yr net vs **+10.8%/yr** for buy-and-hold -- it **trails by -2.46%/yr**. Sitting out "frothy" years mostly means sitting out good years. Nothing to capture. |
| **Sample size** | ![tiny-n](https://img.shields.io/badge/tiny--n%20(41%20yrs)-8b949e?style=flat-square) | Only 41 forecastable annual observations; even the "working" sub-periods are the kind of fishing the desk exists to flag. |

> **In one sentence:** the bell-at-the-top folklore confuses a *coincident* euphoria gauge for a *leading* one -- IPO volume is high *during* good years (+0.32 same-year correlation) but tells you essentially nothing about next year's market (HAC *t* = -0.07), and a long/flat rule built on it underperforms buy-and-hold by ~2.5%/yr.

## The claim

> *Is a flood of IPOs the bell at the top?*

## What we tested

We pair the canonical annual US IPO count (Jay Ritter / University of Florida --
operating companies, ex-SPACs/REITs/CEFs/ADRs/units/penny) with the S&P 500
calendar-year **price** return (Shiller index level, dividends excluded). We
standardize the IPO count with an expanding, past-only window (the "froth
z-score"), then regress the **next-year** S&P return on it with Newey-West HAC
errors -- a one-year execution lag, no look-ahead. We separate the
*contemporaneous* correlation (froth vs same-year return) from the *predictive*
one (froth vs next-year return), sweep three sub-periods to expose the
subsample-mining trap, and race a long/flat froth-timing rule against
buy-and-hold net of costs. A deterministic synthetic positive control confirms
the regression recovers a froth slope when one is planted.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the bell-at-the-top story, the famous tops (1999, 2021), and why the timing never lines up one year ahead -- in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the HAC predictive regression, contemporaneous-vs-predictive correlation gap, sub-period subsample-mining warning, the long/flat-vs-buy-and-hold race, and the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`ipo_volume/`](ipo_volume/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
