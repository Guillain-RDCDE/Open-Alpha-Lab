# Study 936 — Tolerance Bands ⚖️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do bands beat the calendar? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | No specification clears the bar. 5/25 bands minus annual is **−0.026** excess Sharpe (HAC *t* on the return difference **+0.82**, paired bootstrap CI **[−0.067, +0.014]**), **−0.028** on the 3-asset book, and **−0.032** over 2003-2026 *gross-of-cash* against a **−0.034** gross-of-cash comparator on the headline window — same non-result in both eras, at 0–50 bps, and across 2/10→10/50 band widths. Daryanani's reported ~0.5 pp/yr edge shows up here as **+0.09 pp/yr**. Sleeves are the surviving mega-ETFs (SPY/IEF/GLD/BIL) picked with hindsight, but survivorship cannot rescue a null. |
| **Tradability** — is it bankable? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Nothing to bank: every schedule trades **7–17% of NAV a year**, so even a punitive 50 bps one-way separates them by **under 4 bp/yr** of drag. The CIs are tight, not wide — this is a measured nothing, not an underpowered test. **Tax is not modelled** and runs *against* the higher-turnover schedules. |
| **Does the schedule control risk?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | Yes, and here bands do win — on one statistic. SPY stayed inside **53.6–65.2%** under 5/25 for **27% less traded notional than quarterly**, while the do-nothing book finished at **84.6% equity**. But annual is cheaper still (0.079 traded/yr, at a 41.8–67.2% envelope) and quarterly tracks target more closely *on average* (mean abs dev 0.013 vs 0.018). Bands minimise the **worst excursion**. Governance, not return. |

> **In one sentence:** switching a 60/40 (or 50/30/20) book from calendar rebalancing to 5/25 tolerance bands changes the risk-adjusted outcome by about **0.03 of a Sharpe point in the wrong direction and no basis points that survive a *t*-test** — what bands genuinely buy is the tightest *worst-case* weight envelope for less turnover than quarterly, which is a risk-governance win, not an edge.

## What we tested

The Swedroe/Bogleheads **5/25 rule** at full strength: *"trade when a sleeve is off target
by 5 percentage points absolute or 25% relative — better risk-adjusted returns than the
calendar, for less turnover."* One book, four schedules — **drift**, **annual**,
**quarterly**, **5/25 bands** — on **60/40 SPY/IEF** and **50/30/20 SPY/IEF/GLD**, total
return, one execution lag (breach seen at the close of *t*, traded at the close of *t+1*),
one-way cost x traded notional, **excess-of-cash** vs BIL, 2007-05-30 → 2026-06-30 plus a
2003-2026 SPY/IEF cross-check run **gross-of-cash** (BIL predates 2007 — cash cancels in
the return difference but *not* in a Sharpe difference, so that window is compared only
against a gross-of-cash comparator). HAC *t* on the daily return difference, paired
block-bootstrap CIs on the Sharpe difference, an era cut, a 0–50 bps cost sweep, and a
sweep of the band width itself (it is an **assumption**, not a calibration). **Assumptions
labelled and swept:** the 5/25 widths, the 5 bps cost, the target mixes; **tax is not
modelled at all**. **Distinct from [Study 97](../97-balancing-act/)** (is 60/40 better than
100% equity — an *allocation* question, schedule held fixed) and
**[Study 102](../102-free-rebalance/)** (is there a *rebalancing bonus* vs letting the book
drift). 936 takes both as settled and asks the next question down: **which trigger**.
Also distinct from **[Study 604](../604-month-end-rebalancing-flows/)**, which trades
*other people's* rebalancing flows.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the schedule debate feels important, the 0.03-Sharpe answer, the 84.6%-equity book that nobody chose, what bands actually buy |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the excess-of-cash race, HAC return-difference *t*, paired bootstrap CIs, era cut, cost and band-width sweeps, the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run (returns fp `958f652ce4c9`, as-of 2026-06-30): [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`rebal_bands/`](rebal_bands/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
