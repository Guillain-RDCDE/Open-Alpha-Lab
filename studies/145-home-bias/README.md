# Study 145 — Home-Bias

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Blend Sharpe **0.515** < US-only **0.591**; Sharpe Δ = −0.077, bootstrap 95% CI **[−0.165, +0.012]** (95% of resamples favour US-only); mean excess HAC *t* = −1.17. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Blend underperforms US-only on return (−0.96%/yr), vol (+0.89pp), Sharpe (−0.077), and max drawdown (−2.9pp) — at **zero cost**. Nothing to trade here. |
| **Diversification benefit?** | ![Correlation_killed_it](https://img.shields.io/badge/Correlation_killed_it-8b949e?style=flat-square) | SPY-EFA mean rolling correlation **0.83** (max 0.97); US outperformed EFA by **+3.1%/yr**. Both levers ran against the free-lunch claim. |

> **In one sentence:** international diversification (60/25/15 SPY/EFA/EEM) underperformed US-only on every axis over 2003-2026 because high cross-market correlations (0.83 mean) and US structural outperformance (+3.1%/yr) together cancelled the textbook free lunch.

## What we tested

The received wisdom of portfolio construction: US investors concentrate too much in domestic equity and should diversify internationally for a "free lunch" of better risk-adjusted return via low correlation with foreign markets. We test this literally — a static **60% SPY / 25% EFA / 15% EEM** portfolio, rebalanced annually, from 2003 (EEM inception) to 2026 — against 100% SPY as the honest baseline. We measure Sharpe ratio, drawdown, the block-bootstrap CI on the Sharpe difference, and the realised rolling correlation to diagnose *why* the diversification benefit does or doesn't materialise. The study is not a future prediction (international CAPE may now be more attractive than US CAPE); it is an honest historical accounting of what the advice delivered in this period.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the free-lunch claim, why it requires low correlation, the actual correlations, and the cumulative return comparison in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stat on mean excess return, block-bootstrap Sharpe CI, crisis-vs-calm correlation decomposition, and a synthetic positive control sweeping correlation |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`home_bias/`](home_bias/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
