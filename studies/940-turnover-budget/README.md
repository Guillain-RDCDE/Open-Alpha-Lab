# Study 940 — The Turnover Budget ⏱️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the edge real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Over 26.5 years the sleeve's best **gross** excess-of-cash HAC *t* across all four speeds is **+0.67** (Sharpe +0.121 daily → +0.013 quarterly). Every bootstrap Sharpe CI straddles zero; no speed-vs-speed race clears \|*t*\| = 2; a seven-point parameter neighbourhood tops out at *t* = +0.55. The one arm that looks significant — long-only, *t* ≈ **+2.6** vs cash — is **equity beta**: against a **cost-matched, same-clock** equal-weight-11 control its selection alpha is +0.67% to −0.31%/yr with every \|*t*\| ≤ 0.35. |
| **Tradability** — is it bankable? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The honest budget is **2.5 bps** per unit traded notional at daily, 4.8 weekly, 10.1 monthly and **negative** at quarterly — a budget for a return that is statistically zero. At 5 bps + 40 bps borrow three of four speeds are net-negative. The ranking of speeds **inverts below 1 bp**, and the era cut swaps the winner. |

> **In one sentence:** Running one cross-sectional sector-momentum sleeve at four rebalance speeds prices the folk theorem exactly — a daily clock trades **31× NAV a year** against a monthly clock's 5.8× for a gross Sharpe edge of just +0.012 (*t* = +0.18) — but the budget it buys is worthless, because the gross return funding it is indistinguishable from zero at every speed.

## What we tested

Rank the eleven **Select Sector SPDRs** on **12-1** total return (252 days, skipping 21),
long the top 3 / short the bottom 3, equal-weighted, dollar-neutral at 1.0 gross. The *only*
thing that varies is the rebalance clock — **daily / weekly / monthly / quarterly**. One
execution lag (signal through `t`, weights effective `t+1`, trade charged `t+1`); weights
drift between rebalances and are renormalised so the book never silently levers; costs are
one-way × NAV on traded notional; the short leg pays borrow. The deliverable is each speed's
**break-even cost per unit of traded notional**, plus a cost surface, a borrow sweep, a
parameter neighbourhood, a 2013 era cut, bootstrap CIs, paired speed races and a long-only
beta check against a **cost-matched** equal-weight control, over 1999-12-23 → 2026-06-30
(total-return closes, `auto_adjust=True`). **Dedup:** distinct from **28-carousel** and
**225-sector-rotation** (which ask whether a rotation *signal* pays; we hold the signal fixed
and price the clock), **836-timing-luck** (varies the rebalance *day*, not the frequency),
**102-free-rebalance** (no cross-sectional signal), **141-turnover-anomaly** /
**821-turnover-volatility** (turnover as a stock *characteristic*, not the book's own
trading), and **890-sector-risk-parity** / **903-sector-neutral-lowvol** (same eleven ETFs,
weighting schemes rather than a timed sort).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "rebalance more often" is a bill, what 31× turnover a year actually means, the beta trap, the honest verdict |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the frequency table, break-even costs, the cost surface and its sub-1 bp inversion, borrow sweep, era cut, bootstrap CIs, paired races, the cost-matched beta check, the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`turnover_budget/`](turnover_budget/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
