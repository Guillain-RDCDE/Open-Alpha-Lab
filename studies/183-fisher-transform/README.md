# Study 183 — Fisher-Transform

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Gross **-2.59 bps/trade**, HAC *t* = **-0.93** at 5-day hold; 1-day *t* = **-2.29** (significant *loser*); no instrument \|*t*\| > 1.3. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Gross edge is already negative at every tested holding period; costs compound a pre-existing loss with no break-even. |
| **Transform adds info?** | ![No--monotone](https://img.shields.io/badge/No--monotone-8b949e?style=flat-square) | Fisher and raw-x crossovers coincide on **100%** of signal bars across all six tickers — the atanh wrapper is invariant to the crossover operator. |

> **In one sentence:** Ehlers' Fisher Transform is monotone in the normalised price, so it cannot add information to the crossover signal beyond the raw price extremes; measured honestly vs a random-direction coin on six daily equity tapes over 10 years, it delivers a noise-level 5-day result and a statistically significant *loss* at 1-day, burying the "sharper turning points" claim under both a mathematical proof and 8,329 real trades.

## What we tested

Ehlers (2002) maps the close's position within the rolling high-low range through the atanh (Fisher) function — producing a "more Gaussian" oscillator — and trades its crossover with its one-bar-lagged trigger. We test this on **six liquid daily tickers** (SPY, QQQ, IWM, AAPL, TSLA, NVDA) over a **10-year window**, pinning the signal against a **random-direction control** on identical entries and sweeping **five holding periods** (1, 3, 5, 10, 20 days). We also prove the core claim empirically: replacing Fisher with the raw normalised price produces **identical crossover bars** (100% coincidence), confirming the transform adds zero signal information. A deterministic synthetic tape with tunable AR(1) structure serves as the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the Ehlers claim, the monotonicity proof in plain language, the fair bet vs a coin, the holding-period sweep |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC *t* per instrument and per holding period, bootstrap Sharpe CI, the formal monotonicity test, cost sweep, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`fisher_transform/`](fisher_transform/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
