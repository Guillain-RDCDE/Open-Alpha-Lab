# Study 131 — Utilities-Canary

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Q1−Q5 21-day spread **+87.8 bps/month** but HAC *t* = **+1.58** — below the bar of 2 at every horizon; non-monotone quintile pattern. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Timing overlay (cash in top-20% alert periods) lifts Sharpe 0.43→0.47 via *volatility reduction*, not alpha; spread mean = −0.34 bps/day (*t* = −0.45); 486 switches destroy any net edge. |
| **Coincident or forecast?** | ![Coincident](https://img.shields.io/badge/Coincident--not--forecast-8b949e?style=flat-square) | XLU outperformance *co-occurs* with market stress but does not statistically *precede* weak SPY returns at any actionable horizon. |

> **In one sentence:** the XLU/SPY relative-strength canary is a real coincident signal of defensive rotation during stress, but it does not clear the statistical bar as a *leading* indicator of SPY returns, and the higher Sharpe from going to cash in alert periods is risk-reduction (less beta), not exploitable alpha.

## What we tested

The folk claim: when XLU outperforms SPY on a rolling relative-strength basis (rising 20-day
XLU/SPY log-ratio, the "canary singing"), equities are at elevated risk and forward SPY returns
will be below average — go defensive or to cash. We take this literally: sort trading days
by the rolling percentile rank of XLU/SPY relative-strength momentum into five quintiles (Q1
= SPY outperforming; Q5 = XLU strongly outperforming) and test whether there is a **monotone
descent** in 1-day, 5-day, and 21-day forward SPY returns from Q1 to Q5, with HAC inference
on the Q1−Q5 spread. We also run a binary timing overlay (go to cash in the top-20% alert
bucket) vs unconditional buy-and-hold, sweep costs at the rule's natural turnover, and run a
synthetic positive control to confirm the machinery can detect a planted signal. Real tape:
XLU and SPY daily (1998-12-23 to 2026-06-12, n = 6,909 days, ~26.5 years).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the folk recipe, the quintile results in plain language, why the higher Sharpe isn't alpha, the coincident vs forecast distinction |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stats per quintile and horizon, the spread t-stat, regime Sharpe decomposition, cost sweep, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`utilities_canary/`](utilities_canary/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
