# Study 68 — All-Weather ⛅

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does risk parity diversify for real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Yes: best Sharpe **0.92** (vs 0.81 for 60/40, 0.65 for SPY) and smallest drawdown **−17%** (vs −31% / −55%). |
| **Tradability** — can you actually run it? | ![Investable](https://img.shields.io/badge/Investable-2ea44f?style=flat-square) | Four cheap ETFs, monthly inverse-vol rebalance — a sensible, low-drama core allocation. |
| **"The all-weather portfolio that beats everything"?** | ![Overstated](https://img.shields.io/badge/Overstated-8b949e?style=flat-square) | Unlevered it returned **+6.7%/yr, ~half of SPY's +12.6%**; the edge is risk-adjusted, and needs leverage + a bond bull to close the gap. |

> **In one sentence:** risk parity (the "All-Weather" idea) genuinely diversifies — best Sharpe and a drawdown a third of equities' — so it's real and investable; but the claim that it *beats everything* is overstated, because unlevered it returns half of plain stocks and the leverage needed to catch up rests on a bond bull that 2022 ended.

## What we tested

The **risk-parity / All-Weather** allocation: weight assets by *risk* rather than dollars — inverse to each one's volatility — so no single market dominates. We build it **unlevered** on four cheap ETFs (**SPY / IEF / GLD / DBC** — stocks, Treasuries, gold, commodities), rebalance monthly, and compare its Sharpe, volatility and drawdown to equal-weight, a 60/40 mix, and plain equities over 2006–2026. The offline control is a synthetic multi-asset world where assets share a Sharpe but differ in volatility (and a null where they don't).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a smoother ride isn't the same as a bigger return |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the four allocations, the Sharpe/drawdown win, the leverage & bond-bull caveats |

The fingerprinted real-data run (SPY/IEF/GLD/DBC, 2006–2026, fp `b68a69dca595`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (reads the shared cross-asset pull); the offline machinery proof runs on the synthetic world in [all_weather/data.py](all_weather/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
