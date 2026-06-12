# Study 16 — Storm-Shy 🌊

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does scaling exposure by inverse vol actually pay? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Volatility is the one thing markets *do* forecast — realized-variance AR(1) ρ ≈ **+0.66 / +0.74 / +0.66** on SPY/QQQ/EFA. Sizing by its inverse lifts the net Sharpe (**+0.65→+0.76** SPY, **+0.51→+0.77** QQQ; EFA is flat at **+0.43→+0.44**) with a Moreira–Muir spanning alpha that is **HAC-significant on SPY and QQQ** (*t* = 2.4 / 3.5; EFA 0.8) and a circular-block-bootstrap Sharpe-gain CI clearing zero on QQQ (**[+0.07, +0.45]**). **The flag we will not bury:** on SPY the CI is **[−0.06, +0.30]** (P(gain<0) = 10.8%) and on EFA **[−0.18, +0.22]** — **SPY alone would not clear the bar; the stamp rests on QQQ plus the breadth of the published literature**, with SPY directionally supportive and EFA a candid miss on the Sharpe half. |
| **Tradability** — does it survive costs, capacity, scale? | ![Investable](https://img.shields.io/badge/Investable-2ea44f?style=flat-square) | The desk's **first green**: turnover ~**9×/yr** (a slow vol forecast ⇒ a few bps vs a far-above break-even), and the instrument is an **index** — capacity in the hundreds of billions, the way real vol-target & risk-parity books run. Drawdowns collapse on *all three* tapes (SPY **−55%→−39%**, QQQ **−83%→−38%**, EFA **−61%→−31%**), and the gain is **not a tuned point**: across a 10/21/63-day window × 8–15% target grid it spans just [+0.11, +0.13] on SPY and [+0.22, +0.26] on QQQ. The rolling 5y Sharpe gain, measured on the *real* tapes, shows **no post-publication cliff** (QQQ 2020s mean +0.17, latest window +0.20; SPY oscillates around a small positive with its best decade the 2020s). |
| **Free lunch?** | ![Risk-managed](https://img.shields.io/badge/Risk--managed-8b949e?style=flat-square) | No — and we won't pretend. The gain needs **leverage in calm regimes**, and a matched-risk CRRA certainty-equivalent (Cederburg et al.) shrinks it to a smaller, real number (≈ +2.1%/yr SPY, +6.9%/yr QQQ, +0.3%/yr EFA). Risk management, not alpha from nothing. |

> **In one sentence:** after fifteen mirages, the one that holds — because it forecasts **risk** (which is predictable) instead of **returns** (which aren't): size by inverse volatility and you harvest a scalable, decades-stable lift in risk-adjusted return that is statistically certified on QQQ, directionally supportive on SPY (whose CI alone straddles zero — we say so in bold), carried everywhere by a drawdown collapse that needs no significance test, and honestly bounded by the leverage it demands.

## What we tested

The residual that kept surfacing as the *only* real thing inside the desk's fancier teardowns — the **vol-targeting overlay** [Study 12](../12-paper-prophet/) caught hiding inside an ARIMA+GARCH stack — promoted to the lead. The steelman, at full strength (Moreira & Muir, *"Volatility-Managed Portfolios"*, **Journal of Finance** 2017): because expected returns barely move with recent volatility while volatility itself is strongly persistent, scaling exposure by `σ_target / σ̂` — a *past-only* position, no return forecast — raises the Sharpe and earns a spanning alpha. We run it through the full protocol on real SPY (since 1993), QQQ (since 1999) **and EFA (developed ex-US, since 2001)** daily total-return closes — a third, non-US tape so the verdict can't be a one-market artefact — and, keeping the desk honest on its first win, price the gain *and* its bound with a certainty-equivalent test at matched risk, interval it with a **circular block bootstrap** (i.i.d. resampling understates the uncertainty on a vol-clustered tape), sweep the vol-window × vol-target grid (no magic point), and measure the rolling 5-year Sharpe gain on the real tapes (no decay cliff).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story in plain language: why volatility clusters, why dodging storms beats predicting prices, and the drawdowns that vanish |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the full machinery: variance-forecastability AR(1), the Moreira–Muir spanning alpha with HAC errors, a bootstrap Sharpe-gain CI, the CRRA certainty-equivalent at matched risk, and the flat-vol null |

The real run — every fingerprinted, as-of'd SPY/QQQ/EFA number, including the parameter sweep and the rolling-5y decay check — is in [docs/results.md](docs/results.md); the **beat-7 worked complement** (the *no-borrowing test* — cap leverage at 1.0 and the de-risk slice still carries ~100% of the edge, so a leverage-constrained book keeps the benefit) is in [docs/extension.md](docs/extension.md). Reproduce offline via [examples/run_synthetic_demo.py](examples/run_synthetic_demo.py); on the real tape via [examples/verify.py](examples/verify.py) and [examples/extension.py](examples/extension.py) (`--fetch` once to populate the daily-close cache).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
