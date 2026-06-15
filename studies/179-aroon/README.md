# Study 179 -- Aroon

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Pooled HAC *t* = **+3.25**, bootstrap CI entirely positive -- but driven by the **long side only** (short *t* = +0.29) and only IWM is individually significant. Cross-sectionally fragile. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | ~8.5 trades/yr -- costs are trivial, but the long-only skew and small sample (n=381 over 15 years) create substantial model risk. |
| **Beats a coin?** | ![Partially](https://img.shields.io/badge/Beats_a_coin%3F-Partially-dab617?style=flat-square) | The crossover beats a random-direction control by **+52 bps** (+3.25 vs -1.24) -- but only on the long leg. Shorts are coin-like. |

> **In one sentence:** Chande's Aroon(25) crossover shows a statistically real but fragile long-side signal on US ETFs (+36 bps/trade, pooled t=3.25, surviving Bonferroni correction), undercut by cross-sectional inconsistency and a 15-year bull-market window that flatters every long-biased rule.

## What we tested

Tushar Chande's 1995 Aroon indicator locates where, within a 25-bar look-back, the most
recent highest-high and lowest-low fall.  Aroon-Up near 100 = recent high is recent = uptrend
developing; Aroon Oscillator crossing above 0 = same signal.  We test the canonical rule on
**three US-equity ETFs** (SPY, QQQ, IWM) over **15 years of daily bars**, entering at the
next open after each crossover and holding for 5 days.  The honest baseline is a
**random-direction control** on the same entry dates, so the verdict is: does Aroon *direction*
add anything beyond random timing of the same market exposure?

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what Aroon detects, the crossover in plain language, the long/short asymmetry, the cost reality |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-instrument HAC *t*, bootstrap Sharpe CI, hold-sweep, cost sweep, directional breakdown, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`aroon/`](aroon/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
