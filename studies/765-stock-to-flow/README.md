# Study 765 — Stock-to-Flow 📐

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the S2F valuation gap predict BTC returns? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Out-of-sample, the residual (price vs the frozen model line) predicts forward returns at **no** horizon with \|t\| ≥ 2 (best HAC *t* = **−1.68**, 180d). The full-sample "significance" is just the fit talking to itself — and the celebrated R² ≈ 0.9 is spurious: `ln(SF)` is **96%** correlated with the calendar, and a plain price-on-time trend fits equally well (**0.876 vs 0.880**). |
| **Tradability** — can you trade the model? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | A buy-when-cheap-vs-model timer returns **+328%** net vs buy-and-hold's **+1355%** out-of-sample (and loses on Sharpe, 0.71 vs 0.91), landing *below* the 95th percentile of a matched-exposure random-timing placebo. Post-2021 it collapses to buy-and-hold. |
| **Holds out-of-sample?** | ![Busted](https://img.shields.io/badge/Holds_out--of--sample%3F-Busted-8b949e?style=flat-square) | Coefficients frozen at publication: R² falls **0.705 → 0.213**; the model implied a six-figure BTC floor straight through the 2022 crash to **$17k**, and stands **~4×** above the tape by 2026 (**$247k** model vs **$59k** actual). |

> **In one sentence:** PlanB's Stock-to-Flow model fits Bitcoin's history at R² ≈ 0.9 — but so
> does a calendar (`ln(SF)` is 96% a clock), its coefficients frozen at publication miss the next
> seven years and over-predict price four-fold by 2026, and a strategy built on its valuation gap
> loses to simply holding: a textbook spurious regression, not a scarcity law.

## What we tested

In 2019 **PlanB** ([*"Modeling Bitcoin's Value with Scarcity"*](https://medium.com/@100trillionUSD/modeling-bitcoins-value-with-scarcity-91fa0fc03e25))
proposed that Bitcoin's price obeys a power law in its **stock-to-flow ratio** (existing coins ÷
annual new issuance) — reporting an in-sample **R² ≈ 0.95** and famous six-figure price targets
(later, a **$288k** call for the 2024 cycle). We reconstruct the S2F curve *exactly from the
deterministic issuance schedule* (block-reward halvings — not a proxy), pair it with the real
BTC-USD tape, and run the honest test the folklore skips: **freeze the fit at the publication
date and project forward**, then ask whether the valuation gap is a tradable signal net of costs
vs buy-and-hold, with a placebo and a synthetic positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a 95%-fit model can know nothing, the six-figure predictions vs the crash, and why "buy when cheap" lost to holding |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the spurious-regression race, the frozen out-of-sample R² collapse, the HAC residual→return regression, the timer-vs-HODL backtest and the 20-seed synthetic null |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`stock_to_flow/`](stock_to_flow/). The S2F curve is reconstructed from Bitcoin's exact
issuance schedule (consensus law, not an estimate); BTC-USD is a single-survivor asset, named.
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
