# Study 271 — Cardboard-Box

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Forecast slope (box growth → next-year S&P) is **wrong-signed** and insignificant: HAC **t = −1.61** (rail **−1.81**); coincident **t = −0.07**; R² **4.5%**. No \|t\| ≥ 2 anywhere on the real tape. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A "long when boxes grow" timer earns **6.5%/yr net** vs **8.2%/yr** buy-and-hold; its hit-rate (61.8%) is *below* the unconditional up-rate (74.6%). |
| **Busted?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Box and rail output are a fine *coincident* GDP gauge — they dip in every recession — but sold as a *leading* market signal they carry no forecasting content. |

> **In one sentence:** "Dr. Cardboard" describes the economy well and forecasts the market not at all — a coincident gauge mistaken for a leading one, with a year-ahead slope that is small, insignificant, and (if anything) points the wrong way.

## What we tested

The folklore: because nearly everything ships in a corrugated box and rides a freight
train, year-over-year growth in **box (containerboard) shipments** and **rail carloads**
is an early read on the real economy and therefore a leading indicator for stocks. We
hardcode a curated annual box/rail-freight growth series (1970–2024) in `data.py`, join
each year's growth to the **next** year's ^GSPC calendar-year **price** return (a one-year
forecast lag — no look-ahead), and run a Newey-West (HAC) predictive regression for both
predictors. We separate the *coincident* same-year link from the *leading* one, then run a
cost-charged timing backtest (10 bps one-way on NAV, shorts not used) against buy-and-hold.
A synthetic positive control confirms the regression detects a forecasting beta when one is
planted; the real tape has none.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the recession-tracking chart, the forecast scatter, the timing backtest in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | Newey-West predictive regression, coincident-vs-lead split, cost-charged backtest, power check, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`cardboard_box/`](cardboard_box/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
