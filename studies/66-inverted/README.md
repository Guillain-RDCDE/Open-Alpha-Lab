# Study 66 — Inverted 📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does an inverted curve forecast weak equities? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Strongly. After the 10y−3m curve inverts, the next **18 months of equities averaged +1.0% vs +15.8%** normal; the next **24 months −8.7% vs +21.7%** — a 30-point gap. |
| **Tradability** — can you time the market with it? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Inversions are rare (~5% of months), the lead is **long and variable** (6–24m), and near-term markets often rise (12-month gap only −4.2%). |
| **"Forecasts equity downturns"?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | The best recession barometer there is (Estrella-Mishkin) — for *odds and posture*, not a switch on a date. |

> **In one sentence:** the inverted yield curve genuinely forecasts weak equity returns at the 18–24-month horizon (+1% / −9% vs +16% / +22% normally) — a real, economically huge macro signal — but it's a slow, rare, variable-lag predictor that markets often climb into for a year first, so it sets the odds, it doesn't time the market.

## What we tested

The famous macro rule that an **inverted yield curve** (short rates above long rates) forecasts recession — and, by extension, weak equity returns. We compute the **10-year minus 3-month** Treasury slope over 1985–2026 and compare the S&P 500's forward return after an *inverted* curve vs a *normal* one, across 12-, 18- and 24-month horizons — the recession lead is long, so the horizon matters. The offline control is a synthetic world where a mean-reverting, occasionally-inverting curve forecasts forward equity returns (and a null).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "the curve inverted" is a real warning — and a terrible sell button |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the conditional forward returns by horizon, the strengthening gap, the variable-lead caveat |

The fingerprinted real-data run (^TNX/^IRX/^GSPC, 1985–2026, fp `e68e9ffd8fbd`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (`--fetch` to download); the offline machinery proof runs on the synthetic world in [inverted/data.py](inverted/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
