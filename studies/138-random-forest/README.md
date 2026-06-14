# Study 138 — Random-Forest

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | OOS accuracy 53.5% vs shuffled-label control 52.6% (gap = +0.83pp); the shuffled model's gross t-stat (2.34) **exceeds** the real model (2.15) — both harvest SPY beta, not a learned directional signal. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Net Sharpe +0.20 at 5 bps vs buy-and-hold +0.86; the strategy underperforms a passive index fund and collapses to negative at 10 bps. |
| **Beats shuffle?** | ![No](https://img.shields.io/badge/No-8b949e?style=flat-square) | The shuffled-label control matches or outperforms the real model on every gross metric — the decisive test that a noise-trained model can replicate "alpha". |

> **In one sentence:** a Random Forest trained on standard SPY technical features (lagged returns, RSI, volatility, momentum, volume ratio) in a strict walk-forward produces out-of-sample accuracy indistinguishable from a shuffled-label control, confirming the "ML alpha" story is in-sample overfitting dressed up as a strategy.

## What we tested

A staple of retail quant tutorials: train a Random Forest on daily technical features and trade the direction signal on SPY.  We take it seriously: strict walk-forward (504-day training window, 63-day test blocks, refitted each quarter), a shuffled-label control as the primary falsification (identical architecture, training labels permuted — if the shuffled model scores similarly to the real one, the features are useless), and a cost sweep.  The model is deliberately capable (300 trees, max depth 6) so in-sample performance looks real.  A synthetic positive control confirms the machinery detects genuine learning when a signal is planted.  Distinct from Study 39 (MLP on crypto): tree ensemble on equities, explicit feature-importance transparency.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the ML pitch, why in-sample accuracy is misleading, the walk-forward vs shuffle comparison in plain English, costs and buy-and-hold |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stats, shuffled-model gross Sharpe comparison, beta-vs-alpha decomposition, cost sweep, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`random_forest/`](random_forest/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
