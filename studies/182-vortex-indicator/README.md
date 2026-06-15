# Study 182 — Vortex-Indicator

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Best pooled HAC *t* = **+1.63** (20-day hold), below the inference bar of 2 and well below the Bonferroni-adjusted bar of 2.58 (5 hold periods tested). No instrument clears |*t*| = 2. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | ~22 crossovers/ETF/yr: expected annual gross ≈ 4–4.4%/yr on an unestablished edge; costs leave it intact but the signal does not earn its statistical keep. |
| **Beats a coin?** | ![Mixed](https://img.shields.io/badge/Beats_a_coin%3F-Mixed-dab617?style=flat-square) | Cross beats the random control in this sample (70% of coin seeds fell below), but the cross itself does not reach 2σ — not certifiable. |

> **In one sentence:** the Vortex Indicator's VI+/VI− crossover shows a consistent but statistically sub-threshold positive bias on 10 years of SPY/QQQ/IWM daily bars (best HAC *t* = 1.63, below the 2.0 inference bar), making it a weak echo of the trend — not a certified edge.

## What we tested

The Vortex Indicator (Botes & Siepman, *Technical Analysis of Stocks & Commodities*, January 2010) builds two directional movement lines — VI+ (upward vortex) and VI− (downward vortex) — by normalising a 14-bar rolling sum of range-crossing movements by the sum of true ranges. A VI+/VI− crossover is used as a trend-change signal: go long when VI+ crosses above VI−, short when VI− crosses above VI+. We run the crossover as a fixed-horizon forward-return backtest (hold = 5/10/14/20/30 days) across three liquid ETFs (SPY, QQQ, IWM, 10 years of daily data), pin it against a **random-direction control** on identical entry bars, apply a Bonferroni multiple-comparisons correction across the five hold horizons, and sweep costs at ~22 trades/yr. A deterministic synthetic tape with tunable bar-level momentum serves as the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the VI+ / VI− idea explained simply, the coin comparison in plain language, why the sub-threshold *t* matters |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-instrument HAC *t*, the hold-period sweep with Bonferroni correction, cost sweep, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`vortex_indicator/`](vortex_indicator/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
