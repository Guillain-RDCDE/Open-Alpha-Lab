# Study 199 — Sales-Growth

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Hedge -0.4%/yr, HAC *t* = -0.17 -- far below the |t| >= 2 bar, wrong direction. On the survivor-biased S&P 500 panel, high-growth "glamour" stocks barely outperform low-growth "value" stocks: the LSV (1994) effect is absent. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No gross edge exists. Annual rebalancing of large-cap names is cheap in principle, but there is no alpha to harvest -- the hedge is statistically zero. |
| **Survivorship bias?** | ![Upper_bound_only](https://img.shields.io/badge/Survivorship-Upper_bound_only-8b949e?style=flat-square) | Panel is current S&P 500 members projected back. High-growth failures are absent. Even this best-case result shows no effect. |

> **In one sentence:** the Lakonishok, Shleifer & Vishny (1994) prediction that high-growth "glamour" stocks underperform low-growth "value" stocks finds no support on the S&P 500 survivor panel from 2008-2025 -- the hedge is -0.4%/yr with HAC t = -0.17, a clean null in the wrong direction.

## What we tested

Lakonishok, Shleifer & Vishny (1994) argue that investors over-extrapolate past sales
growth, bidding up "glamour" stocks (fast-growing revenues) and neglecting "value" stocks
(slow-growing revenues). We compute one-year trailing revenue growth (Revenues_y /
Revenues_{y-1} - 1) from the shared EDGAR Revenues cache, sort ~326 current S&P 500
survivors into quintiles, lag fundamentals by one full year (fiscal year y -> calendar
year y+1 returns), and test whether the low-growth quintile outperforms the high-growth
quintile vs an equal-weight market and a random-portfolio control. A deterministic
synthetic tape with tunable LSV premium serves as the positive control.

The panel is **survivorship-biased**: it covers only firms that remain in the S&P 500 as
of 2026. Even under this best-case bias, the hedge is zero. The original LSV effect was
strongest in small-cap, low-coverage stocks -- the opposite of this panel. **Distinct from
Study 44 (Growth-Spurt)**: that study uses *total-asset* growth (Cooper et al. 2008);
this study uses *revenue* growth, directly testing LSV's overextrapolation mechanism.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the LSV recipe in plain English, why the S&P 500 is the wrong test, year-by-year results, the survivorship-bias caveat |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | quintile return monotonicity (absent), HAC t-stats, random-portfolio null distribution, synthetic positive control sweep |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`sales_growth/`](sales_growth/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
