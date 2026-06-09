# Study 22 — Crystal-Ball 🔮

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there an edge? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The causal (tradable) HP-cycle book earns essentially nothing — on real SPY a Sharpe of **+0.11** (Newey–West *t* = **+0.7**), and similar across **6** ETFs. The spectacular two-sided result is artefact. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The two-sided signal needs *all future prices* to compute **today's** position — it has no live counterpart at all. There is nothing to execute. |
| **Look-ahead bias?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | On a pure random walk the two-sided book "earns" a Sharpe near **2** (*t* > **10**), and its cycle correlates **−0.40** with the *next 5-day return* it was built from. Caught in the act. |

> **In one sentence:** a respectable detrending filter, used innocently, manufactures a Sharpe-2 backtest out of a coin flip — because the classic HP filter is two-sided and the cycle quietly contains the future; recompute it causally and nothing remains.

## What we tested

The desk's fifth idea from Kakushadze & Serur, *151 Trading Strategies* (strategy **§8.1**, moving averages with a Hodrick–Prescott filter) — and its first study about a **backtest trap** rather than a market effect. The steelman: HP-detrend a price and trade the cycle's mean reversion (long when price is below its smooth trend), which backtests beautifully. The catch we expose: the textbook HP filter is **two-sided** — the trend at day *t* uses the *entire* series — so the cycle silently encodes future prices. We prove it on a synthetic **random walk** (where no edge can exist), confirm with a direct perturbation test and a future-return correlation, then show the same fabricated edge — and its disappearance under a causal filter — on real ETFs.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story in plain language: a Sharpe-2 curve on a random walk, the cycle that "predicts" the future, and the edge that vanishes the moment you stop peeking |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the machinery: the perturbation proof, two-sided vs one-sided Sharpe with HAC *t*, the future-leakage correlation, and the λ/cost robustness that *passes* (the danger) |

The real run — every fingerprinted, as-of'd ETF number — is in [docs/results.md](docs/results.md); the **beat-7 worked complement** (the bias *survives* every λ and cost — only causality kills it) is in [docs/extension.md](docs/extension.md). Reproduce offline via [examples/run_synthetic_demo.py](examples/run_synthetic_demo.py); on the real tape via [examples/verify.py](examples/verify.py) and [examples/extension.py](examples/extension.py) (`--fetch` once to populate the close cache).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
