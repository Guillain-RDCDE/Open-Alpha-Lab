# Study 266 — Misery-Index

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**Does the misery index (inflation+unemployment) call equity returns?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Forward-return slope on the misery level: β = **+0.1pp** per 1-SD, **t_HAC = +0.08**; on the misery *change*, **t_HAC = −0.23**. High-minus-low tercile spread **−0.2pp** (Welch t = −0.06, p = 0.95). Nothing on a 76-year, serially-correlated tape. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The "buy when misery is high" contrarian timing rule earns **2.7%/yr** net vs **8.2%/yr** buy-and-hold — it sits out 64% of years and forfeits the equity premium. |
| **Busted?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Both the leading-bear and the maximum-pessimism contrarian stories fail: the misery print has no out-of-sample content for next-year S&P returns. |

> **In one sentence:** the misery index is a fine description of how bad the economy *feels*, but it carries no forward information about the stock market — neither the level nor the change predicts next year's S&P return at any meaningful significance.

## What we tested

The misery index (Arthur Okun) is `CPI year-on-year inflation + unemployment rate`. We hardcode the December CPI-YoY (BLS CPI-U) and December U-3 unemployment rate for 1948–2025 in `data.py`, join the December misery print of year *Y* to the S&P 500 calendar-year price return of year *Y+1* (a full-year, no-look-ahead execution lag), and test two competing stories: **leading-bear** (high misery → low forward returns) and **contrarian/maximum-pessimism** (high misery → high forward returns). We regress the forward return on the standardized misery level and on its year-on-year change, judging significance by a **Newey-West (HAC) t-stat** (overlapping macro regimes are serially correlated). We add a high-minus-low tercile sort, a contrarian timing backtest (gross and net of one-way costs × NAV), and a synthetic positive control that confirms the engine detects a planted misery→return beta. Returns are **price-only** (no dividends); the S&P is **survivorship-clean** (the index itself, not a stock basket), so the Signal verdict is not inflated by survivorship.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the misery timeline, the base-rate trap, the contrarian "buy the bottom" idea tested in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC predictive regressions on level and change, the tercile sort, the timing backtest net of costs, the n=76 power calculation, the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`misery_index/`](misery_index/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
