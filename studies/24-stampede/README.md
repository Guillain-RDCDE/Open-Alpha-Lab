# Study 24 — Stampede 🐂

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do past winners keep winning? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Overwhelming in the long-run literature and on our synthetic control (a WML alpha at HAC *t* ≈ **14**) — but on the modern S&P 500 the 12-1 winners-minus-losers factor earns an alpha of just **+4.4%/yr** (HAC *t* = **+0.9**), indistinguishable from zero. Momentum has *decayed* in US large caps. |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Thin standalone Sharpe (**+0.10**), fast turnover (**~7×/yr**), and you must *short* the beaten-down losers; long-only winners (Sharpe **+0.99**) barely beat the equal-weight market (**+0.97**). |
| **Crash risk?** | ![Severe](https://img.shields.io/badge/Severe-8b949e?style=flat-square) | The signature momentum crash: a worst month of **−22.5%** and a max drawdown of **−61%** — when crushed losers violently rebound, the short leg detonates. Forecastable, though: vol-scaling cuts the drawdown to **−32%**. |

> **In one sentence:** the most robust anomaly in finance is real in principle and on our control, but faint on the modern large-cap sample and weighed down by a catastrophic — if forecastable — left tail.

## What we tested

The desk's seventh idea from Kakushadze & Serur, *151 Trading Strategies* (strategy **§3.1**, price momentum). The steelman, at full strength (Jegadeesh & Titman, *"Returns to Buying Winners and Selling Losers"*, **Journal of Finance** 1993): rank stocks by their trailing 12-1 return, go long the top decile (winners) and short the bottom (losers), refresh monthly, and the spread earns a large, significant premium across markets and decades. We prove the apparatus on a synthetic panel with a *baked-in* persistent relative drift (and a no-momentum null that must — and does — earn nothing), then run the WML factor on the current S&P 500 cross-section and split "is the premium real?" from "can you live with how it pays?".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story in plain language: winners that keep winning, a premium that's gone faint, and the crash that tramples you when the herd turns |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the machinery: the decile forward-return profile, the WML CAPM alpha with HAC errors, the crash skew and drawdown, the sub-sample decay, and risk-managed momentum |

The real run — every fingerprinted, as-of'd S&P 500 number — is in [docs/results.md](docs/results.md); the **beat-7 worked complement** (the *risk-managed momentum* overlay — vol-scaling tames the crash, cutting the drawdown from −61% to −32%) is in [docs/extension.md](docs/extension.md). Reproduce offline via [examples/run_synthetic_demo.py](examples/run_synthetic_demo.py); on the real tape via [examples/verify.py](examples/verify.py) and [examples/extension.py](examples/extension.py) (`--fetch` once to populate the shared S&P 500 panel cache).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
