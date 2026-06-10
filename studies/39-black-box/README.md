# Study 39 — Black-Box 🤖

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The gorgeous **in-sample** Sharpe (BTC **5.38**, 66% accuracy) collapses to a coin-flip out-of-sample: walk-forward OOS accuracy **0.51** across all four coins (BTC/ETH/LTC/XRP, 3448 days), the small gross OOS Sharpe just a long-bias riding the bull. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | There is no directional OOS edge for costs to erode; the book flips often and goes net-negative at realistic crypto costs (ETH/LTC/XRP already negative by 10 bp; BTC −0.18 at 20 bp). |
| **Overfit?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | The net memorises **randomly shuffled** labels in-sample at **0.66–0.69** training accuracy — identical to the **0.664** on true labels. The in-sample number is fitting capacity, not skill. |

> **In one sentence:** an MLP fed crypto OHLCV scores a dazzling in-sample Sharpe (5.38) that collapses to a 51% coin-flip out-of-sample and turns negative after costs — the in-sample edge was the net memorising noise, proven by a shuffled-label control, exactly the backtest trap of [Study 22 (Crystal-Ball)](../../22-crystal-ball/).

## What we tested

Kakushadze & Serur's *151 Trading Strategies* §18.2 catalogues **neural-network cryptocurrency trading**:
feed a multilayer perceptron a small bank of causal, price-derived features (lagged returns, momentum,
realised vol, a mean-reversion z-score) and let it predict the **sign of tomorrow's return**. The
believer's case is that young, retail-driven crypto markets are full of nonlinear patterns a net can
find — and in-sample, the net certainly looks like it does. We run the only honest test there is —
**walk-forward** out-of-sample (fit on the past, predict the unseen next block, repeat) — on real daily
BTC/ETH/LTC/XRP, with a synthetic positive control (a planted weak signal the net *should* recover) and a
random-walk null. It is the machine-learning cousin of [Study 22 (Crystal-Ball)](../../22-crystal-ball/),
the desk's other study about a **backtest trap** rather than a market effect.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a beautiful in-sample backtest is the easiest way to fool yourself, and how walk-forward exposes it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | in-sample vs walk-forward Sharpe & accuracy, the cost wall, the shuffled-label control, the deflated-Sharpe caveat |

The fingerprinted real run is in [docs/results.md](docs/results.md); the beat-7 shuffled-label
overfitting control is worked in [docs/extension.md](docs/extension.md).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
