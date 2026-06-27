# Study 507 -- Cross-Sectional-Momentum

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

> The most replicated anomaly in finance -- buy past winners, short past losers. Does the
> canonical 12-1 sort still pay on a small modern survivor basket?

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- do past winners keep winning? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Quintile WML earns **-0.34%/yr**, HAC *t* = **-0.072**, label-shuffle placebo *p* = **0.545** -- indistinguishable from a coin. The decile sort nudges to +1.24%/yr (*t* = +0.20, *p* = 0.38), still flat. **Survivorship is named on the signal axis**: the basket is names *still trading in 2026*, so the loser leg's natural shorts (firms that trended to delisting) are absent -- these flat numbers are already an upper bound. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Best-case net return **+0.02%/yr** (decile, after 10bps/leg + 50bps borrow), with a **-53.5%** drawdown and **30%/mo** turnover. The quintile is negative net. Nothing to trade. |
| **Decile vs quintile?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Tightening the sort did not surface an edge -- it swapped a slightly-negative number for a slightly-positive one inside the noise band, on thinner 4-name legs with a deeper drawdown. Concentration does not rescue momentum here. |

> **In one sentence:** the canonical Jegadeesh-Titman 12-1 momentum factor -- overwhelming in the
> literature -- earns nothing (HAC *t* near zero, placebo *p* > 0.5) on a 38-name large-cap
> survivor basket where every name was a decade-long winner, and no amount of decile concentration
> brings it back.

## What we tested

Jegadeesh & Titman (1993): each month, rank the basket by its trailing 12-month return *skipping
the most recent month* (the classic "12-1", which dodges short-term reversal), go long the top
fraction (winners) and short the bottom (losers), equal-weight, dollar-neutral, hold one month.
We prove the apparatus on a deterministic synthetic panel with a *baked-in* relative-strength
drift (and a no-momentum null that earns nothing), then run the winners-minus-losers (WML) book
on **38 large-cap survivor names** (yfinance daily adjusted-close, 2012--2025, 151 holding
months) at **both** the quintile (top/bottom 20%) and decile (top/bottom 10%) cut. One execution
lag (form on the month-end close, hold the next month -- no same-bar fill, no look-ahead); costs
of 10bps/leg/rebalance plus a 50bps/yr short borrow; a label-shuffle placebo null; survivorship
named on the **signal** axis as an upper-bound caveat.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | momentum in plain language, why a basket of survivors makes everyone a winner, the synthetic control, the flat real result, and the honest verdict |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the 12-1 signal, the WML construction, HAC inference, the label-shuffle placebo, decile-vs-quintile, year-by-year crashes, equity curve and drawdown, gross-vs-net |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`cross_sectional_momentum/`](cross_sectional_momentum/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
