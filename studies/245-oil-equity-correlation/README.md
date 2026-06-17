# Study 245 — Oil-Equity Correlation

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

Does the oil price lead the stock market, or just move with it?

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Weekly predictive t = **−0.46** (near-zero, wrong sign); monthly t = **+0.15**; OOS R² = **−1.8%** — the inference bar is not cleared at any horizon. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No directional forecast signal → no tradable equity-timing edge. The coincident link is real but untradable. |
| **Contemporaneous or Forecast?** | ![Confirmed coincident](https://img.shields.io/badge/Contemporaneous--only-8b949e?style=flat-square) | Contemporaneous R² **9.2%** (t = +4.36) — oil moves *with* equities, not *before* them. |

> **In one sentence:** Oil is a real coincident indicator for equities (contemporaneous R² 9.2%, t = 4.4) but a complete failure as an equity forecaster — the weekly predictive t-stat is −0.46 and the OOS R² is negative, making it an even weaker predictor than Dr. Copper.

## What we tested

The folk belief that crude oil price changes *predict* (not just track) equity returns.
We use USO (United States Oil Fund ETF) and CL=F (WTI crude front-month futures) vs SPY
(S&P 500 ETF), 2006-04-10 through 2026-06-16 (~20 years, 1,052 weekly periods). We run
lagged regressions of weekly oil log-returns on *forward* equity returns (strictly t+1),
testing in-sample (HAC-robust t-stat) and out-of-sample (Goyal-Welch expanding-window OOS
R²), and separate the contemporaneous link (visually real, ~9% R²) from the predictive
link (the thing you'd need to trade — essentially zero).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the oil-leads-stocks story, the contemporaneous vs predictive split in plain language, why the chart fools you |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stats, Goyal-Welch OOS R², DM test, horizon sensitivity (weekly vs monthly), synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`oil_equity_correlation/`](oil_equity_correlation/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
