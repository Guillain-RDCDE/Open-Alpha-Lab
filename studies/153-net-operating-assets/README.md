# Study 153 — Net-Operating-Assets

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | On the survivor-biased S&P 500 panel: hedge +6.6%/yr, HAC *t* = +3.53 — but the panel excludes every firm that failed, so the t-stat is an upper bound. Literature support for the effect on broader universes exists but is contested on large caps. |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Annual rebalancing of ~20 names is cheap; but the alpha estimate is uncertain (upper bound only), the NOA effect is primarily documented for small/mid-caps with higher transaction costs, and the long-short inverts in some recent years. |
| **Survivorship bias?** | ![Upper_bound_only](https://img.shields.io/badge/Survivorship-Upper_bound_only-8b949e?style=flat-square) | Panel is current S&P 500 members projected back. High-NOA failures are absent. Real live edge is unknown and likely smaller. |

> **In one sentence:** balance-sheet bloat (high NOA) appears to predict low future returns on the S&P 500 survivor panel, but the signal is inflated by survivorship bias — the true live edge on large caps is unknown and the academic evidence is strongest for small/mid-cap stocks outside this panel.

## What we tested

Hirshleifer, Hou, Teoh & Zhang (2004) argue that firms accumulating large operating-asset surpluses — balance-sheet "bloat" — earn low future returns because investors over-extrapolate past accounting-based earnings growth. We compute NOA = (Operating Assets − Operating Liabilities) / lagged Total Assets, where Operating Assets = Total Assets − Cash and Operating Liabilities = Total Liabilities − Long-Term Financial Debt. All four concepts are in the shared EDGAR cache. We sort the current S&P 500 survivors into quintiles on this signal, lag fundamentals by one full year (fiscal year y → calendar year y+1 returns), and test whether the low-NOA quintile outperforms the high-NOA quintile vs an equal-weight market and a random-portfolio control. A deterministic synthetic tape with tunable NOA premium serves as the positive control.

The panel is **survivorship-biased**: it covers only firms that remain in the S&P 500 as of 2026. Every headline number is an upper bound.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the NOA recipe in plain English, the survivor-bias caveat, year-by-year results, why large caps are the wrong testing ground |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | quintile return monotonicity, HAC t-stats, random-portfolio null distribution, synthetic positive control sweep |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`net_operating_assets/`](net_operating_assets/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
