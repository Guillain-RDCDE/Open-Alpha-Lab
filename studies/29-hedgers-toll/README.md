# Study 29 — Hedgers-Toll 🌾

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does hedging pressure predict commodity returns? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Real and strong on our synthetic control (information coefficient HAC *t* > **11**) and in the long-run literature — but on the modern CFTC-COT + futures tape the IC is **−0.021** (*t* = **−1.4**) and the top-minus-bottom spread is *negative* (**−7.6%/yr**). |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The dollar-neutral hedging-pressure factor *loses money*: Sharpe **−0.41** (Newey–West *t* = **−1.4**), negative across **every** signal-normalisation window, at **5.1×/yr** turnover. Long-only-top Sharpe **+0.16** vs the equal-weight basket **+0.67** — you'd have been better off just owning commodities. |
| **Hedging-pressure premium today?** | ![Faded](https://img.shields.io/badge/Faded-8b949e?style=flat-square) | Sub-period Sharpes **−0.83 / −0.90 / +0.56**: negative through the bulk of 2015–2025 (a late bounce aside). A structural risk premium that a present-day trader cannot collect on the liquid commodities. |

> **In one sentence:** the toll producers supposedly pay speculators for absorbing their hedges is real in the long-run record and on our control, but on the modern, liquid commodity futures the booth is empty — the hedging-pressure factor loses money at every parameter.

## What we tested

The desk's twelfth idea from Kakushadze & Serur, *151 Trading Strategies* (strategy **§9.2**, trading on hedging pressure). The steelman (Keynes' *normal backwardation*; Cootner 1960; Gorton, Hayashi & Rouwenhorst, *"The Fundamentals of Commodity Futures Returns"*, **Review of Finance** 2013): when commercial hedgers are heavily net short, speculators take the long side and earn a risk premium — read straight off the CFTC Commitments-of-Traders report. We prove the engine on a synthetic commodity panel where hedging pressure predicts returns by construction (and a null), then run the dollar-neutral hedging-pressure factor on **real CFTC COT positioning** (legacy futures-only, 12 commodities) against **Yahoo commodity futures** — both free, both fetched once and cached.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story in plain language: who pays the toll, why a structural premium is so appealing, and the empty booth |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the machinery: the cross-sectional IC, the long-short factor with HAC errors, the comparison to the commodity basket, and the window sweep |

The real run — every fingerprinted, as-of'd COT/futures number — is in [docs/results.md](docs/results.md); the **beat-7 worked complement** (the factor is negative at *every* signal window — an absence, not a tuning miss) is in [docs/extension.md](docs/extension.md). Reproduce offline via [examples/run_synthetic_demo.py](examples/run_synthetic_demo.py); on the real tape via [examples/verify.py](examples/verify.py) and [examples/extension.py](examples/extension.py) (`--fetch` to download CFTC COT + commodity futures).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
