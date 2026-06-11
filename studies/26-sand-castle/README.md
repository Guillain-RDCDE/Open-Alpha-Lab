# Study 26 — Sand-Castle 🏖️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the reversion real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | A stretched-up residual really does bounce: cross-sectional information coefficient **+0.008** (*t* = **+2.4**), and *gross* of cost the daily mean-reversion book earns a Sharpe of **+0.61**. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | It trades **every day** (~180×/yr turnover), so net of a realistic cost the Sharpe is **−2.14** — deeply under water. A real, tiny daily edge living entirely inside transaction costs. |
| **Does optimization help?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The §3.18 selling point — mean-variance optimization (`w ∝ C⁻¹E`) — *backfires*. The sample covariance is near-singular (condition number **1.7×10¹⁷**), so `C⁻¹` amplifies estimation noise, and the optimized book (net **−2.14**) is **worse** than the naive signal-weighted book (net **−0.44**). Even shrinkage only claws it back *toward* naive, never past it. |

> **In one sentence:** a genuine but minuscule daily reversion, harvestable only on paper — and the mean-variance optimizer wrapped around it is a sand castle: it inverts a noisy covariance, amplifies the noise, and ends up worse than not optimizing, before the daily turnover washes the whole thing away.

## What we tested

The desk's ninth idea from Kakushadze & Serur, *151 Trading Strategies* (strategy **§3.18**, statistical arbitrage via optimization). The steelman: given expected stock returns `E` and a covariance `C`, the Sharpe-maximizing dollar-neutral portfolio is `w ∝ C⁻¹E` — so an "optimal" stat-arb book accounts for the correlations a naive signal-weighting ignores. Here `E` is a short-horizon residual mean-reversion signal (Lehmann 1990; Lo–MacKinlay 1990). We prove the engine on a synthetic panel whose residual mean-reverts by construction (and a no-reversion null), then run the daily, causal optimized and naive books on the current S&P 500 — and ask whether inverting an estimated covariance helps or, as Michaud warned, *maximizes the estimation error*.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story in plain language: a real but tiny daily bounce, the optimizer that makes it worse, and the cost that washes it all away |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the machinery: the cross-sectional IC, optimized-vs-naive net Sharpe, the covariance condition number, gross-vs-net, and the shrinkage sweep |

The real run — every fingerprinted, as-of'd S&P 500 number — is in [docs/results.md](docs/results.md); the **beat-7 worked complement** (does covariance *shrinkage* rescue the optimizer? — it only converges it to naive) is the shrinkage table in the same [docs/results.md](docs/results.md). Reproduce offline via [examples/run_synthetic_demo.py](examples/run_synthetic_demo.py); on the real tape via [examples/verify.py](examples/verify.py) and [examples/extension.py](examples/extension.py) (`--fetch` once to populate the shared S&P 500 panel cache).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
