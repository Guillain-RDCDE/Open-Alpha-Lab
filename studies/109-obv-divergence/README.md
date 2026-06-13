# Study 109 — OBV-Divergence

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | OBV trend gross **−1.87 bps/signal**, HAC *t* = **−0.47**; OBV divergence **−9.80 bps**, *t* = **−1.48**. No instrument or signal clears \|*t*\| ≥ 2. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Both signals start gross-negative; any transaction cost makes them significant losers vs the passive 5-day drift (+22.88 bps, *t* = +3.65). |
| **Volume precedes price?** | ![Not--Supported](https://img.shields.io/badge/Volume_precedes_price%3F-Not_Supported-8b949e?style=flat-square) | OBV neither times the market nor predicts reversals at a credible level across 16 years of daily data on six liquid US names. |

> **In one sentence:** Granville's "volume precedes price" — tested as the OBV trend signal and OBV-vs-price divergence reversal on 16 years of liquid daily data — produces nothing beyond a coin flip, and the divergence signal is actually sub-random, fighting the unconditional equity drift.

## What we tested

Granville (1963) coined On-Balance Volume and the slogan "volume precedes price": when OBV rises while price lags, informed accumulation is underway, and price will follow. We steelman this in two forms: **(A)** OBV above its 20-day SMA = go long, below = go short (the trend-following version); **(B)** OBV rising while price falls (or vice versa) predicts a price reversal in the divergence direction (the contrarian version). Both run on six liquid US names (SPY, QQQ, IWM, AAPL, MSFT, NVDA), daily bars, 2010–2026 (4,136 days), 5-day forward return horizon, versus a random-direction control on identical signal dates and the buy-and-hold passive baseline. Multiple-comparison-aware (12 tests, Bonferroni threshold |t| ≥ 3.0).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what OBV is, the two recipes, why "volume precedes price" fails in plain English, the coin comparison |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-instrument HAC *t*, the random-direction control, divergence sub-signal breakdown, cost sweep, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`obv_divergence/`](obv_divergence/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
