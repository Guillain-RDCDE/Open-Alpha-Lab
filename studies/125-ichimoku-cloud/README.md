# Study 125 — Ichimoku-Cloud

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Gross **−8.08 bps/trade**, HAC *t* = **−0.94**; indistinguishable from zero. Outperforms a random entry at the same bars by +11.86 bps, but both are negative — the filter reduces the damage, it doesn't produce an edge. Every instrument |*t*| < 1.1. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Negative gross at ~10 trades/year per instrument; no positive break-even cost exists. Low turnover keeps cost drag small but there is no positive budget to absorb it. |
| **Beats a coin?** | ![Partial](https://img.shields.io/badge/Beats_a_coin%3F-Partial-8b949e?style=flat-square) | The signal beats a random entry at the same bars (+11.86 bps), but that only shows that Ichimoku entry points are adverse timing moments — the cloud *reduces* how negative you are, it does not flip the sign. |

> **In one sentence:** the Ichimoku Kinko Hyo composite signal — cloud position plus Tenkan/Kijun cross — is a statistically indistinguishable-from-zero drag on real daily tapes; the famous multi-confirmation filter adds value over blind entry at the same moments, but those moments are already adverse, and the net result is a slightly-less-bad coin flip with no tradable edge.

## What we tested

The Ichimoku Kinko Hyo system (Hosoda 1969) on daily bars, standard settings: Tenkan(9), Kijun(26), Senkou B(52), cloud displacement 26 bars. The canonical folk rule: go **long** when price is above the Kumo cloud (both Senkou A and B) and Tenkan > Kijun; go **short** when below the cloud and Tenkan < Kijun. We steelman this as "a multi-lagged trend-confirmation filter designed to avoid whipsaw — does it generate a positive expected return?" We test with symmetric ±1 ATR(20) exits (the only direction-fair payoff) across four liquid daily tapes (SPY, QQQ, IWM, AAPL; ~10 years), pin it against a **random-direction control** on identical entry bars, sweep costs, and check fixed-day forward returns. No look-ahead: the cloud at bar *t* is computed from data up to bar *t − 26* only.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the cloud in plain language, the two-confirmation trap, the fair bet vs a coin, why the cloud fires at the wrong moments |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-instrument HAC *t*, random-control distribution, fixed-day forward returns, cost sweep, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`ichimoku_cloud/`](ichimoku_cloud/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
