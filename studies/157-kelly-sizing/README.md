# Study 157 — Kelly-Sizing

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | Full-Kelly geometric return nominally exceeds buy-and-hold by ~0.5%/yr, but the HAC *t*-stat on the daily differential is only **1.83** (|*t*| < 2), below the REAL bar. The maths is right; the precision is not. |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Full-Kelly produces an **−85% max drawdown** vs −55% for buy-and-hold. True Kelly fraction for SPY (~3.5x) instantly hits the 2x leverage cap. Even at the cap, Sharpe (0.30) is far below buy-and-hold (0.48). Estimation risk alone can invert the advantage. |
| **Beats buy-and-hold on Sharpe?** | ![No--on--SPY](https://img.shields.io/badge/Beats_BH_on_Sharpe%3F-No-8b949e?style=flat-square) | Buy-and-hold Sharpe 0.48; full-Kelly Sharpe 0.30; half-Kelly Sharpe 0.38. Kelly trades Sharpe for raw geometric growth — and even that marginal growth lift is statistically WEAK. |

> **In one sentence:** the Kelly criterion is mathematically optimal given the true return parameters, but on real SPY data those parameters are too imprecisely estimated to deliver more than a statistically weak (+0.5%/yr) geometric return edge over buy-and-hold, while inflicting an 85% drawdown — making it FRAGILE in practice despite being REAL in theory.

## What we tested

The Kelly criterion states that the growth-optimal leverage fraction for a risky asset is `f* = mu/sigma^2` (continuous, Gaussian case). The claim: size positions by this formula, and in the long run you will outgrow any fixed-fractional strategy. We test this honestly on SPY daily total returns (1993–2026, 33 years) using a strict walk-forward (5-year training window, monthly rebalance, 1 bp cost): compare full-Kelly, half-Kelly, full-exposure (buy-and-hold), and half-exposure on geometric return, Sharpe, and max drawdown. We also visualise the Kelly parabola and the estimation-error distribution across lookback windows to expose the core problem: the true Kelly fraction for SPY is ~3.5x (immediately hitting any practical leverage cap), and the 1-year estimation-window 90% confidence interval is [0.00, 2.00] — essentially no information.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the Kelly criterion is, why it sounds so compelling, the drawdown trap, and why estimation risk defeats it in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | walk-forward backtest, Kelly parabola, estimation-noise distribution, HAC inference on growth differential, Sharpe comparison, the half-Kelly trade-off |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`kelly_sizing/`](kelly_sizing/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
