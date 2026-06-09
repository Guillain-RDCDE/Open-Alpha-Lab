# Study 12 — Paper-Prophet 🧥

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does ARIMA forecast the *direction* of SPY returns better than a coin? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Walk-forward directional hit-rate **51.98%** (HAC *t* = **+0.85**) across 8,091 days — inside the noise band, and even that tilt is just the up-drift, not skill. The author's own "52–55%" is a coin flip. |
| **Tradability** — does the edge survive once you remove the vol-targeting and charge costs? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Stack Sharpe **+0.17**; the **same GARCH sizing on a constant-long signal** scores **+0.53** — the forecast *subtracts* **0.35 Sharpe** (boot CI [−0.74, +0.07], 95% negative). Alpha vs the vol-managed-SPY factor is **+0.0006%/day** ≈ 0. It *is* vol-targeting; the forecast is a tax. |
| **"Win every single trade"?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Per-day win rate **51.98%** — a coin flip; the headline is mathematically impossible and empirically ~50%. |

> **In one sentence:** the "complete time-series trading stack" is **volatility targeting wearing an ARIMA forecast as a trenchcoat** — the GARCH 1/σ̂ sizing earns a real +0.53 Sharpe (documented managed-beta, not alpha), and bolting the coin-flip forecast on top drags it down to +0.17, so the whole stack is *worse than its own risk layer run naked*.

## What we tested

A viral X/Twitter thread by **Roan (@RohOnChain)** — *How To Build A Time Series Model To Win Every Single Trade* (19 May 2026; ~792 K views) — ships runnable code: a `TimeSeriesTradingSystem` that fits **ARIMA(1,0,1)** on a rolling 252-day window for SPY's direction, layers **GARCH(1,1)** to size positions by volatility (`min(1, 1/σ̂)`), and walks it forward. The author himself concedes the direction is a coin flip and "the GARCH sizing is doing more work." So the whole study is one question: once you subtract the volatility-targeting, **is there any forecasting alpha left at all?** We port his code verbatim, run it cold over the full cached SPY tape (**8,091** graded days, 1993–2026), and decompose it — separating the forecast (ARIMA direction) from the sizing (GARCH 1/σ̂) via a constant-long control that reuses the **identical** σ̂ path. As-of 2026-06-01, fingerprint `0c5568d20239`; every number in [`docs/results.md`](docs/results.md).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story + the stakes in plain language: why the ADF discipline is right, and how the GARCH sizing — not the ARIMA forecast — does all the work |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the full machinery: HAC directional test, the Sharpe decomposition (forecast vs sizing), the alpha-vs-managed-beta regression, the cost sweep, the in-sample-vs-walk-forward inflation control |

Both render inline on GitHub (pre-executed). Reproduce the headline via **[examples/verify.py](examples/verify.py)** → [`docs/results.md`](docs/results.md), and the cross-asset panel (the beat-7 worked complement across eight tailwind/no-tailwind assets) via **[examples/extension.py](examples/extension.py)** → [`docs/extension.md`](docs/extension.md).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
