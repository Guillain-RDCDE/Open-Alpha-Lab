# Study 130 — Vol-Risk-Premium

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | VRP premium t = **+22.9** (REAL, mean +3.7 pp, positive 86% of days); Q5 forward return t = **+2.94** (REAL); but Q5-Q1 spread t = **+1.67** (below the |*t*| ≥ 2 bar) and timer underperforms buy-and-hold. Premium confirmed; equity timing signal is weak. |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Timer Sharpe 0.50 vs buy-and-hold Sharpe 0.55; harvesting the raw VRP requires short-options infrastructure; payoff skew = −0.73, worst 21-day period = −19.7% (COVID-19 March 2020). Crash risk is the structural cost of the premium. |
| **VRP premium?** | ![Real](https://img.shields.io/badge/VRP_premium%3F-Real-2ea44f?style=flat-square) | 33 years of daily data, n = 8,379, mean VIX = 19.5%, mean 21d-RV = 15.9%, mean VRP = **+3.7 pp**, HAC *t* = **22.9**. Option sellers are structurally compensated. |

> **In one sentence:** the variance risk premium (VIX systematically exceeds trailing realised vol) is one of the most statistically robust facts in finance — but using it to time equity positions is weak-to-mixed, and harvesting it directly requires short-options infrastructure and a crash-risk budget that makes it fragile in practice.

## What we tested

A three-part claim well-supported in the academic literature (Carr & Wu 2009, Bollerslev et al. 2009): (1) implied volatility (VIX) exceeds subsequently realised volatility (trailing 21-day SPY RV) on average — the **variance risk premium** (VRP); (2) a high VRP signals a calm, over-priced-fear environment and predicts better forward SPY returns (quintile-sorted, no look-ahead); (3) a binary timer that is long SPY when the VRP exceeds its rolling median beats buy-and-hold. We test all three against ~33 years of daily VIX and SPY data (March 1993 -- June 2026), using HAC t-statistics throughout. This is distinct from Study 63 (Free-Fall, which shorts the VIXY ETP with roll-cost exposure) and Study 111 (VIX-Term-Structure, which uses the VIX/VIX3M slope slope as a signal).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the premium in plain language, the quintile chart, the timer vs buy-and-hold, the short-vol left-tail skew, why you can not just trade this simply |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stats on each sub-claim, bootstrap Sharpe CI, regime conditioned statistics, short-vol payoff distribution, positive control on synthetic tape |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`vol_risk_premium/`](vol_risk_premium/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
