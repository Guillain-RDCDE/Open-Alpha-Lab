# Study 247 - Bond-Seasonality

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) - see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** - is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | A positive TOM gap is present in both TLT and IEF (6.7 bps/day vs 0.7 bps rest on TLT; 4.6 vs 0.8 bps on IEF), but only IEF clears the HAC *t* = 2 bar (*t* = **+2.66**); TLT falls just short (*t* = **+1.99**). One instrument clearing the bar, one not, with 23 years of data: **Weak** by the inference bar. The turn-of-year effect is **absent** (*t* = -0.40 on TLT, +0.53 on IEF). |
| **Tradability** - does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The TOM-only book is in the market 19% of the time, enough to build a return - but it **underperforms buy-and-hold** in CAGR on both instruments (TLT: **2.81%/yr vs 3.71%** B&H; IEF: **1.96%/yr vs 3.60%**). The whole-tape Sharpe edges higher for IEF (0.66 vs 0.55) due to avoiding mid-month volatility, but the book earns less in absolute dollars. |
| **"Bonds have a reliable calendar bid"?** | ![Mixed](https://img.shields.io/badge/Mixed-8b949e?style=flat-square) | The month-end institutional-rebalancing story is theoretically coherent and shows a suggestive positive gap at the turn of the month. But the evidence is too soft and inconsistent across durations to call it confirmed. The turn-of-year effect is entirely absent. |

> **In one sentence:** there is a **soft, positive turn-of-month pattern in Treasuries** - IEF clears the HAC bar, TLT barely misses - but the TOM-only book **underperforms buy-and-hold** in absolute return on both instruments, making this a **Weak signal and a Mirage to trade**.

## What we tested

The claim at full strength: *"Treasuries have their own calendar - a turn-of-month or
turn-of-year bid driven by institutional rebalancing, coupon reinvestment, and portfolio
flows. Buy TLT (or IEF) at month-end, sell at the start of the next month."* We take it
literally. The **TOM window** is derived from the trading index itself (last 2 + first 2
trading days of each calendar month; no external calendar). We measure TOM vs every other
day (HAC *t*, Wilson win-rate intervals) on **TLT total-return** (20+ yr Treasuries,
2002+) and **IEF total-return** (7-10 yr, 2002+), run a **TOM-only** book against
buy-and-hold, and separately test a **turn-of-year (TOY)** window (last 5 + first 5
trading days per calendar year). A deterministic synthetic tape with a planted TOM bump
(positive control, must be detected) and a flat tape (negative control, must not) is the
harness's spine.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the TOM bar chart, why the bond calendar is softer than equities, the TOM-only book that trails buy-and-hold, the absent turn-of-year |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC *t* on the gap (TLT vs IEF), Wilson win-rate intervals, capacity arithmetic, TOM book vs B&H cumulative equity |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`bond_seasonality/`](bond_seasonality/). **Not investment advice** - research & education. See [LICENSE](../../LICENSE).*
