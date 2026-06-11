# Study 43 — Free-Lunch 🍽️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do low-beta assets beat the market? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Barely. On a tradable ETF cross-section the beta-neutral BAB book's **gross** Sharpe is **0.47 — below SPY's 0.59**. The tilt exists but doesn't clear the simplest benchmark. |
| **Tradability** — does it survive the leverage it needs? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No. Beta-neutrality requires running the low-beta leg at **2.78×**; charge realistic financing and the Sharpe falls **0.47 → 0.20 (3%) → 0.02 (5%)** — at 5% it compounds *negatively*. |
| **"Free lunch"?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The premium **is** the rent on the leverage. Pay the rent (cf. [Study 30 House-Edge](../30-house-edge/)) and the lunch is gone. Gross edge also decayed: Sharpe **0.70 (1999–2012) → 0.28 (2013–on)**. |

> **In one sentence:** betting against beta is sold as a low-risk free lunch, but on a tradable cross-section it doesn't even beat the market *gross*, it needs ~3× leverage to be market-neutral, and a realistic financing rate halves and then erases the edge — the lunch was the leverage bill all along.

## What we tested

The "low-risk anomaly": low-beta assets earn more per unit of risk than high-beta ones, and the **BAB factor** (Frazzini & Pedersen 2014; [paperswithbacktest](https://github.com/paperswithbacktest/awesome-systematic-trading) lists it at Sharpe `0.594`) harvests it by going long low-beta (levered up to beta 1) and short high-beta (levered down). We build it on a liquid-ETF cross-section spanning the beta spectrum, 2000–2026, and ask the question the headline Sharpe quietly skips: **what happens when you pay for the leverage the low-beta leg requires?** We sweep the financing rate, compare to simply owning the market, and split the sample for decay. The offline control is a synthetic factor world with known betas and a tunable low-beta premium (and a null).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "safe stocks beat risky ones" is true *and* not a free lunch, and where the borrowed money goes |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the beta-neutral construction, the 2.78× leverage, the financing sweep that erases the Sharpe, gross-vs-market, the decay |

The fingerprinted real-data run (13 ETFs, 2000–2026, fp `dbdf824e8421`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (`--fetch` to download); the offline machinery proof runs on the synthetic factor world in [free_lunch/data.py](free_lunch/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
