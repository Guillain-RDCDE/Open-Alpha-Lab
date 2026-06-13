# Study 98 — High-Noon 🏔️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Forward returns after an all-time high are statistically *indistinguishable* from forward returns when not near a high. The at-minus-not difference never clears the bar: HAC *t* = **−1.34 / −0.36 / +0.38** at 1 / 3 / 12 months — and at 12 months the high is actually mildly *higher* (+1.4 pts). |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | "Avoid the highs" **trails buy-and-hold by ~1.7 pts/yr** (9.12% vs 10.82% CAGR, net of 5 bps/switch), with a *lower* Sharpe and **no** drawdown relief (−54.7% vs −55.2%). You end with **0.60×** the wealth — you sat out the best trends. |
| **Is an ATH a sell signal?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The opposite of the folklore: 12-month forward win-rate is **84.6%** at-ATH vs **80.5%** not-at-ATH (Wilson intervals don't overlap). Highs cluster in uptrends, and uptrends keep trending. |

> **In one sentence:** buying at an all-time high is **not** the riskiest moment — forward returns after a record are as good as (and at a one-year horizon, slightly *better* than) buying off a high — so the rule *"never buy the high"* is **busted**, and acting on it (sitting in cash 29% of the time) simply costs you **~1.7 points a year** for no risk reduction at all.

## What we tested

A piece of folk wisdom stated at full strength: *"Never buy at an all-time high — it's the riskiest moment, a classic sell signal; wait for a pullback."* We take it literally on **SPY (total return)**: flag every day whose close is within **1%** of the **running all-time high**, then compare the forward **1 / 3 / 12-month** returns of those days against days **not** near a high — means with **HAC *t*-stats**, win-rates with **Wilson** intervals, and a HAC test of the **difference**. Then the rule as a strategy: an **avoid-the-highs** timer (invested only when *not* at an ATH, cash at 0% otherwise, act one day later, **5 bps**/switch) versus buy-and-hold. A deterministic synthetic tape with a planted *mean-reversion-at-highs* knob is the positive control — the harness flags highs as bearish there, and correctly *doesn't* on a driftful random walk.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the rule, what *actually* happens after a record high, the win-rate bars, why "wait for a pullback" left money on the table |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | conditional forward-return means with HAC *t* and Wilson CIs, the HAC test of the difference, the avoid-highs backtest vs buy-and-hold |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`high_noon/`](high_noon/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
