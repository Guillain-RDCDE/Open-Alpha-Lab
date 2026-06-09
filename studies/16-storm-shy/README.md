# Study 16 — Storm-Shy 🌊

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does scaling exposure by inverse vol actually pay? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Volatility is the one thing markets *do* forecast — realized-variance AR(1) ρ ≈ **+0.66 / +0.74** on SPY/QQQ. Sizing by its inverse lifts the net Sharpe (**+0.65→+0.76** SPY, **+0.51→+0.77** QQQ) with a Moreira–Muir spanning alpha that is **HAC-significant** (*t* = 2.4 / 3.5) and a bootstrap Sharpe-gain CI clearing zero on QQQ. |
| **Tradability** — does it survive costs, capacity, scale? | ![Investable](https://img.shields.io/badge/Investable-2ea44f?style=flat-square) | The desk's **first green**: turnover ~**9×/yr** (a slow vol forecast ⇒ a few bps vs a far-above break-even), and the instrument is an **index** — capacity in the hundreds of billions, the way real vol-target & risk-parity books run. Drawdowns collapse (SPY **−55%→−39%**, QQQ **−83%→−38%**). |
| **Free lunch?** | ![Risk-managed](https://img.shields.io/badge/Risk--managed-8b949e?style=flat-square) | No — and we won't pretend. The gain needs **leverage in calm regimes**, and a matched-risk CRRA certainty-equivalent (Cederburg et al.) shrinks it to a smaller, real number (≈ +2.1%/yr SPY, +6.9%/yr QQQ). Risk management, not alpha from nothing. |

> **In one sentence:** after fifteen mirages, the one that holds — because it forecasts **risk** (which is predictable) instead of **returns** (which aren't): size by inverse volatility and you harvest a real, scalable, decades-stable lift in risk-adjusted return, honestly bounded by the leverage it demands.

## What we tested

The residual that kept surfacing as the *only* real thing inside the desk's fancier teardowns — the **vol-targeting overlay** [Study 12](../../12-paper-prophet/) caught hiding inside an ARIMA+GARCH stack — promoted to the lead. The steelman, at full strength (Moreira & Muir, *"Volatility-Managed Portfolios"*, **Journal of Finance** 2017): because expected returns barely move with recent volatility while volatility itself is strongly persistent, scaling exposure by `σ_target / σ̂` — a *past-only* position, no return forecast — raises the Sharpe and earns a spanning alpha. We run it through the full protocol on real SPY (since 1993) & QQQ (since 1999) daily total-return closes, and — keeping the desk honest on its first win — price the gain *and* its bound with a certainty-equivalent test at matched risk.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story in plain language: why volatility clusters, why dodging storms beats predicting prices, and the drawdowns that vanish |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the full machinery: variance-forecastability AR(1), the Moreira–Muir spanning alpha with HAC errors, a bootstrap Sharpe-gain CI, the CRRA certainty-equivalent at matched risk, and the flat-vol null |

The real run — every fingerprinted, as-of'd SPY/QQQ number — is in [docs/results.md](docs/results.md); the **beat-7 worked complement** (the *no-borrowing test* — cap leverage at 1.0 and the de-risk slice still carries ~100% of the edge, so a leverage-constrained book keeps the benefit) is in [docs/extension.md](docs/extension.md). Reproduce offline via [examples/run_synthetic_demo.py](examples/run_synthetic_demo.py); on the real tape via [examples/verify.py](examples/verify.py) and [examples/extension.py](examples/extension.py) (`--fetch` once to populate the daily-close cache).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
