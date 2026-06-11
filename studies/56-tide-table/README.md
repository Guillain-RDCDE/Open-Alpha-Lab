# Study 56 — Tide-Table 🌊

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does CAPE forecast long-run returns? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Yes, strongly. CAPE's correlation with the next-10-year real return is **−0.51** (R² **0.28**); cheap-CAPE decades returned **+10.2%/yr** real vs **+4.0%** for expensive ones — a monotone valuation ladder over 116 years. |
| **Tradability** — can you time the market with it? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Not really. At a *1-year* horizon R² collapses to **0.05** — an "expensive" market can keep rising for a decade (1995–2000, 2015–2021). It sets expectations, it doesn't time. |
| **"Forecasts long-run returns"?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | The valuation signal the [Fed Model](../../47-paper-moon/) only pretended to be — valuation carries the signal, the bond comparison didn't. |

> **In one sentence:** the Shiller CAPE genuinely forecasts the next decade's real returns (R² ~0.28, a monotone +10% / +4% cheap-to-expensive ladder) — a *real* and valuable signal — but it's a tide table, not a stopwatch: at a one-year horizon it's almost useless, so it calibrates expectations rather than timing the market.

## What we tested

The **cyclically-adjusted P/E (CAPE)** — price over a 10-year average of real earnings — and whether it forecasts future returns (Campbell & Shiller 1998; the time-series cousin of [paperswithbacktest](https://github.com/paperswithbacktest/awesome-systematic-trading)'s *"Value/CAPE"* entry, Sharpe `0.351`). We run it on **116 years** of Shiller data: CAPE's correlation and R² with the next-10-year *real* return, the valuation-bucket ladder, the implied-return fit — and the honest limit, the same regression at a **1-year** horizon, to show CAPE is a long-run forecaster, not a timer. The offline control is a synthetic world where CAPE forecasts forward returns (and a null).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why valuation tells you the decade ahead but not the year ahead |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the 10y vs 1y R², the valuation-bucket ladder, the implied-return fit, the timing caveat |

The fingerprinted real-data run (Shiller 1900–2016, fp `dc9da4d822f5`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (`--fetch` to download); the offline machinery proof runs on the synthetic world in [tide_table/data.py](tide_table/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
