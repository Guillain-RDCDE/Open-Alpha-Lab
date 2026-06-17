# Study 290 — September-Effect

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**Is September really the market's worst month?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | September lags by **−0.79pp/mo** (mean +0.00% vs +0.79% for other months) and the one-sided permutation flags it (p = **0.034**), but the serial-correlation-robust Newey-West *t* = **−1.93** falls short of |t| ≥ 2, the Welch test is non-significant (p = 0.074), and the drag is concentrated in 1986–2005 (fails persistence). |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Avoiding September gains **+0.07pp/yr gross**, **−0.03pp/yr net** of two trades a year — before tax. No exploitable edge. |

> **In one sentence:** September genuinely is the weakest month on the modern S&P, but the drag is small, statistically borderline once you correct for serial correlation, isn't even the most-negative month (October is), and is worth nothing once you try to trade it.

## What we tested

We compute S&P 500 monthly **price** returns from the Shiller dataset (1950–2025,
912 months, 76 Septembers), isolate September against the **pooled other-month**
baseline, and run a Newey-West (HAC) t-test on the September dummy (the headline),
a one-sided permutation test, a Welch two-sample test, a sub-period persistence
check, and an avoid-September trading backtest charged for two trades a year. A
synthetic positive control confirms the machinery recovers a planted September
drag cleanly; the real tape sits right at the edge of detectability.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the monthly profile, the base-rate view, the avoidance rule, in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t on the Sep dummy, permutation distribution, sub-period persistence, n=76 power, positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`september_effect/`](september_effect/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
