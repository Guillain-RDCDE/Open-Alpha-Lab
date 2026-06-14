# Study 151 — Stocks-For-Long-Run

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Mean 30-year real equity return **+6.66%/yr**, mean 30-year excess over bonds **+4.63%/yr**, HAC *t* = **+30.9** (overlapping windows — effective N ≈ 5; directionally conclusive). |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | The premium requires 20–30 years of patience. At 10 years, **11%** of windows are negative real for equity; bonds won **4 of 15** decades including the entire 2000s. |
| **Useful?** | ![Only_if_you_can_wait_30_years](https://img.shields.io/badge/Only_if_you_can_wait_30_years-8b949e?style=flat-square) | Siegel's 'always beats' claim is an almost-truth: at 20y the worst real equity return was **−0.18%/yr** (one window, 1901); at 30y bonds came within **−0.12%/yr** of equity once (1902). The premium is real; the 'always' is rhetoric. |

> **In one sentence:** Siegel's equity-always-wins claim is historically almost-true — the long-run real equity premium (+6.7%/yr) is one of finance's most robust facts, but it requires 30 years of uninterrupted patience to guarantee positive returns, and a U.S.-specific survivorship lens makes 'always' a rhetorical near-miss rather than an iron law.

## What we tested

The core claim of Jeremy Siegel's *Stocks for the Long Run* (1994–2022): **over any 20- or 30-year period on historical record, U.S. equities have never delivered a negative real return and have never lost to bonds.** We test this literally on Shiller's monthly panel (1871–2023), computing every rolling 1-, 5-, 10-, 20-, and 30-year annualised real equity total return (real price + real dividends reinvested) and real bond proxy (10-year nominal yield minus CPI). We find the worst-case window for each horizon, count the fraction of windows where equity goes negative or loses to bonds, and assess the four headline claims (H₁–H₄). The signal (equity premium) is robustly REAL. The tradability is FRAGILE: the horizon requirement is brutal, the path is not smooth, and the sample is U.S.-survivorship-biased.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim and its near-misses in plain language, the worst-case chart by horizon, the decade-by-decade picture, why 30 years is the minimum |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stats with overlapping-window caveat, H₁–H₄ formal tests, the synthetic positive control, survivorship discussion, the cumulative real TR chart |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`stocks_for_long_run/`](stocks_for_long_run/). Data: Shiller S&P 500 monthly panel (`_cache/shiller_sp500.parquet`). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
