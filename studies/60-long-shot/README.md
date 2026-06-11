# Study 60 — Long-Shot 🎯

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do low-skew commodities beat high-skew? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The right direction, stably: long-low/short-high-skew earns **+5.2%/yr (hit 53%)**, Sharpe **0.28 then 0.27** across the sample — but Sharpe **0.27 (Lo t 1.1)** isn't significant on a 14-name basket. |
| **Tradability** — can you bank it? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Commodity-ETF roll and spreads erode it: net Sharpe **0.19 at 10 bp**, 0.08 at 25 bp. |
| **"Skewness/lottery effect is real"?** | ![Supported](https://img.shields.io/badge/Supported-8b949e?style=flat-square) | Sign + literature (Fuertes-Miffre-Fernandez 2015) agree; the small universe just can't confirm the magnitude. |

> **In one sentence:** the commodity skewness/lottery effect — investors overpay for jackpot-like, positively-skewed commodities, so the low-skew ones out-earn — points the right way and is stable (~+5%/yr), but a Sharpe of 0.27 (t 1.1) over a 14-ETF basket, thinned by costs, can only support it, not confirm it.

## What we tested

The **skewness (lottery) effect in commodities** (Fuertes, Miffre & Fernandez-Perez 2015; [paperswithbacktest](https://github.com/paperswithbacktest/awesome-systematic-trading) lists it at Sharpe `0.482`): positively-skewed, lottery-like commodities are over-bid and underperform, so you go long the low-skew names and short the high-skew. We rank a **14-commodity ETF basket** each month by trailing-12-month return skewness, run the long-short, and measure its sign, significance, persistence and cost-sensitivity. The offline control is a synthetic commodity panel where high-skew assets genuinely underperform (and a null).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "avoid the lottery commodities" leans the right way — and why a small basket can't prove it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the long-short with its Lo t-stat, the persistence, the cost sweep, the breadth limit |

The fingerprinted real-data run (14 commodity ETFs, 2009–2026, fp `de0a072075bb`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (`--fetch` to download); the offline machinery proof runs on the synthetic panel in [long_shot/data.py](long_shot/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
