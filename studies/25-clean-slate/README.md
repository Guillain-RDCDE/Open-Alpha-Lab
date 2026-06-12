# Study 25 — Clean-Slate 🧼

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is residual momentum real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Strong on our synthetic control (alpha HAC *t* ≈ **16**); on the modern S&P 500 the residual winners-minus-losers factor earns **+5.0%/yr** (HAC *t* = **+1.0**) — a higher point estimate than [Study 24](../24-stampede/)'s total momentum (+4.4%, *t* +0.9), but still indistinguishable from zero. |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Thin standalone Sharpe (**−0.01**), fast turnover (**~12×/yr**), and you must short the losers — but it's the right *platform* for crash management. |
| **Cleaner than total momentum?** | ![Unproven here](https://img.shields.io/badge/Unproven_here-8b949e?style=flat-square) | Two individually insignificant alphas can't certify a difference, so the gap gets a **paired block-bootstrap** on the books' common window: skew gap **−0.19** (95% CI **[−0.88, +0.44]**), Sharpe gap **−0.12** (95% CI **[−0.28, +0.02]**) — neither clears zero, and both lean the *wrong* way. The drawdown does collapse to **−30%** (Sharpe **+0.16**) once vol-management is stacked on top, but that is the overlay's engineering as much as the residual's. |

> **In one sentence:** stripping the market out of momentum raises the alpha point estimate and gives a clean platform for crash control — but under a paired test the "cleaner cousin" advantage doesn't clear zero on this tape, and what reliably tames the crash is the vol-management stacked on top.

## What we tested

The desk's eighth idea from Kakushadze & Serur, *151 Trading Strategies* (strategy **§3.7**, residual momentum) — the natural sequel to [Study 24 (Stampede)](../24-stampede/), which found total-return momentum real-in-principle but faint and crash-prone. The steelman (Blitz, Huij & Martens, *"Residual Momentum"*, **Journal of Empirical Finance** 2011): run 12-1 momentum on each stock's *residual* return — the part not explained by the market/factors — and you keep the premium while shedding the systematic, beta-driven crash. We prove the engine on a synthetic panel where momentum is baked into the residual (and a no-momentum null), then run a **1-factor (market) residual** version on the current S&P 500 — a stated simplification of the source's Fama-French 3-factor residual.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story in plain language: why stripping the market should tame the crash, the faint premium, the paired test the "cleaner cousin" claim fails here, and the defence stack that finally collapses the drawdown |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the machinery: the causal residualisation, the residual-WML CAPM alpha with HAC errors, the null-harness calibration (seed battery), the paired residual-vs-total bootstrap, and the defence stack |

The real run — every fingerprinted, as-of'd S&P 500 number — is in [docs/results.md](docs/results.md); the **beat-7 worked complement** (the *defence stack* — residualise *and* vol-manage, drawdown −67% → −30%) is in [docs/extension.md](docs/extension.md). Reproduce offline via [examples/run_synthetic_demo.py](examples/run_synthetic_demo.py); on the real tape via [examples/verify.py](examples/verify.py) and [examples/extension.py](examples/extension.py) (`--fetch` once to populate the shared S&P 500 panel cache).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
