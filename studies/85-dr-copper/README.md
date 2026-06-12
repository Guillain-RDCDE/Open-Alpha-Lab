# Study 85 — Dr-Copper

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Weekly IS t = **−1.81** (wrong sign); monthly t = **+1.17** (right sign, < 2); OOS R² **−0.30%** — the predictive bar is not cleared. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No reliable directional forecast → no tradable equity-timing edge. The coincident link is real but untradable. |
| **Contemporaneous or Forecast?** | ![Confirmed](https://img.shields.io/badge/Contemporaneous--only-8b949e?style=flat-square) | Contemporaneous R² **12.2%** (t = +6.4) — the ratio moves *with* equities, not *before* them. |

> **In one sentence:** Dr. Copper is a real coincident indicator (contemporaneous R² 12%) but flunks as an equity forecaster — the weekly predictive regression is wrong-signed (t = −1.81) and the OOS R² is negative, confirming Goyal & Welch: what looks predictive in-sample is a historical-mean mirage out-of-sample.

## What we tested

The "copper/gold ratio has a PhD in economics" narrative — popularised by Jeffrey Gundlach and embedded in macro dashboards — claims the Cu/Au ratio *predicts* (not just tracks) equity returns and bond yields because copper demand embeds growth expectations while gold absorbs risk-off flows. We take the strongest testable version: a lagged regression of weekly copper/gold log-ratio changes on *forward* equity returns and yield changes, tested in-sample and out-of-sample (Goyal-Welch expanding-window OOS R²), over 25 years of daily data (HG=F, GC=F, ^GSPC, ^TNX, 2000–2026, 1,344 weekly periods). We separate the contemporaneous link (visually striking, ~12% R²) from the predictive link (the thing you'd need to trade) and find them sharply different.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the PhD story, the contemporaneous vs predictive split in plain language, why the chart fools you, why OOS kills it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stats, Goyal-Welch OOS R², DM test, horizon sensitivity (weekly vs monthly), the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`dr_copper/`](dr_copper/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
