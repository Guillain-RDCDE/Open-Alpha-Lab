# Study 272 — Champagne-Index

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**Does champagne consumption mark the top?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Predictive slope is **negative** (corr **−0.33**) in the folklore direction and **sign-stable** under leave-one-out — but the robust HAC (Newey-West) t-stat is **−1.30** (< 2), and the two-sided permutation p = **0.117**. n = 25 forward years is underpowered to confirm it. Price-only & survivorship named here. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The long/short overlay nets **+4.7%/yr** vs **+6.8%/yr** for buy-and-hold; shorting equities in "party" years fights their upward drift, and costs + borrow finish the job. |
| **Busted?** | ![Mostly](https://img.shields.io/badge/Mostly-8b949e?style=flat-square) | The marquee hits (2007 boom → 2008 crash, 2021 record → 2022 slump) are real and seductive, but a tercile spread on n = 25 dominated by a few crisis years is how mirages are born. |

> **In one sentence:** the champagne points the right way — boomier years are followed by softer markets — but it whispers where a tradable edge would have to shout; the robust slope test lands at |t| = 1.3, and the cost-aware long/short trails just buying the index.

## What we tested

The Champagne Indicator (folklore): when champagne shipments boom, euphoria has peaked
and equities are near a top — so high champagne *growth* should predict *low* forward
equity returns. We hardcode worldwide champagne shipments (CIVC figures, 1999–2024) in
`data.py`, pair each year's YoY growth with the **next** year's ^GSPC price return (the
look-ahead-free *forward* convention — the CIVC publishes the figure in January of the
following year), and run a predictive OLS with a **Newey-West HAC** t-stat on the slope,
a permutation test (10,000 shuffles) on both the slope and a tercile spread, a
leave-one-out robustness check, an n = 25 power calculation, and a cost-aware long/short
backtest with one-way costs and short-borrow. The synthetic positive control confirms the
machinery fires when a real champagne→equity link is planted; the real tape sits below the
detection threshold.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the shipment series, the scatter, the contrarian split, the buy-and-hold comparison, in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | predictive OLS, Newey-West t, permutation distributions, leave-one-out, the n=25 power calculation, the cost-aware long/short |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`champagne_index/`](champagne_index/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
