# Study 20 — Freight-Train 🚂

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the trend predict the future? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Real and strong in the cross-asset academic data and on our synthetic control (pooled time-series-momentum *t* > 3) — but **faint on this equity-heavy 14-ETF menu**: pooled *t* = **+1.5**, strategy Newey–West *t* = **+1.8**, right at the edge of significance. |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | It *survives costs* easily (turnover only **3.3×/yr** — the break-even is far above any realistic level), but standalone it earns a Sharpe of **+0.29**, *below* simply holding the basket (**+0.45**), and it has **decayed**: sub-sample Sharpe **+0.53 → +0.27 → −0.08**. |
| **Crisis hedge?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | The reason you'd actually hold it: in the basket's worst **41** months (basket **−7.83%/mo**) trend *earns* **+0.53%/mo**, at a basket beta of **−0.05** — a "long-volatility", crisis-alpha profile. The standalone Sharpe is a distraction. |

> **In one sentence:** a real but faint, decayed standalone edge that survives costs and earns its keep as a negatively-correlated crisis hedge — held for its *shape*, not its average.

## What we tested

The desk's third idea from Kakushadze & Serur, *151 Trading Strategies* (strategy **§10.4**, futures trend following). The steelman, at full strength (Moskowitz, Ooi & Pedersen, *"Time Series Momentum"*, **Journal of Financial Economics** 2012, across 58 futures and a century of data): an asset's own past-year return predicts its next return, so sizing each position by `sign(R_i^T)/σ_i` across a diversified basket harvests a real, low-turnover edge. We prove the apparatus on a synthetic universe with a *baked-in* persistent drift (and a driftless null that must — and does — kill every leg), then run the §10.4 rule (12-month signal, inverse-vol sized, monthly rebalance) across a basket of 14 diversifying ETFs vs simply holding them.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story in plain language: why trends persist, why the standalone payoff is faint and fading, and why you'd hold it anyway — for the crash months |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the machinery: the pooled time-series-momentum *t*, TSMOM vs the basket, alpha & beta, the sub-sample decay, the crisis-convexity test, and the driftless null |

The real run — every fingerprinted, as-of'd ETF number — is in [docs/results.md](docs/results.md); the **beat-7 worked complement** (the *crisis-convexity test* — what trend does in the basket's worst months, the diversification a thin Sharpe hides) is in [docs/extension.md](docs/extension.md). Reproduce offline via [examples/run_synthetic_demo.py](examples/run_synthetic_demo.py); on the real tape via [examples/verify.py](examples/verify.py) and [examples/extension.py](examples/extension.py) (`--fetch` once to populate the ETF cache).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
