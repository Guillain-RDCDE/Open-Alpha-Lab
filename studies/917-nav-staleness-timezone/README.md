# Study 917 — Stale NAV 🕰️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the edge real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Every next-day slope on SPY is **negative** — the opposite of catch-up. Part is the **domestic control** (SPY on itself, *b* = −0.082, *t* = −3.51) inherited through the fund's US beta, though only *part*: that confound over-explains EWG/EWA/EWU (104–137%) but covers 56% of EWJ and 36% of FXI. Net of it, **no fund is positive anywhere** — the best is EWG at *t* = +1.54, against a five-fund Bonferroni bar of **2.58**. What clears the bar clears it backwards: FXI (*t* = −4.30 unit-hedged, −3.65 at a fitted beta) and EWJ (−3.19, fitted only — the unit hedge is an **assumption** and it hides this), and **both die after 2010** (−1.90, −0.55). Basket timezone-specific trigger-day return **−1.30 bps/day (*t* = −0.39)**, CI [−19.8, +6.6], sign-flipping across eras, **+0.70 bps (*t* = +0.13)** with one more day of delay. |
| **Tradability** — is it bankable? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The rule as stated **loses money**: excess Sharpe **−0.70** net of 10 bps, against **+0.30** for simply holding the same five funds — two spreads a round trip on 10.7% of days, for a session that is if anything slightly below average. The mirror short is the only ever-positive cell (+0.127 **gross**, *t* = +0.70) and is under water by 5 bps one-way, before borrow. Costs are **one-sided** (buy-and-hold is untraded), which runs against the claim, not for it. |

> **In one sentence:** The 1990s stale-NAV trade was real, but its victim was the once-a-day *mutual-fund* NAV strike — a US-listed country ETF prices the US session **as it happens**, so what is left the next day is not a catch-up but over-reaction and reversal, most of it the US market's own, and worth nothing after costs.

## What we tested

Regress each country ETF's day-`t+1` total return on **SPY**'s day-`t` total return (HAC /
Newey-West), for **EWJ, EWG, FXI, EWA, EWU** vs SPY, 1996-03-18 → 2026-06-30 — then trade
it: long an equal-weight basket for the `t+1` return after a **top-decile** SPY day
(expanding quantile, no hindsight), cash otherwise, **one** execution lag, 10 bps one-way ×
NAV, excess-of-cash on every leg (`^IRX` **proxy**, BIL cross-check). Plus the SPY-on-SPY
domestic control and the net-of-SPY slope under **both** hedges — a unit hedge (nothing
fitted, the headline) and a fitted contemporaneous beta (in-sample, a **diagnostic only**,
because the unit-beta shortcut turns out to be load-bearing) — a 2010 era cut with beta
refit inside each era, cost/borrow/threshold sweeps, a conservative extra-lag variant and a
block bootstrap. Cost and borrow are **assumptions**, both swept; the five funds are
**survivors**, which flatters the tape. **Dedup:** distinct from
**379-etf-lead-lag** (an ETF vs its own members, same session), **865** and **870**
(cross-asset and within-sector lead-lag, all US hours), **01-overnight-anomaly** /
**788-overnight-intraday-tug-of-war** (decomposing *one* instrument's own sessions),
**146-country-momentum** (multi-month rotation across these same funds) and **613** /
**916** (the *level* costs of a foreign wrapper — hedge carry, withholding — not its timing).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the trade was once real and went to court, the backwards sign, the confound that explains it, three ways it fails |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-fund HAC slopes raw, unit-hedged and beta-hedged, the confound decomposition, the domestic control, Bonferroni, era cut, bootstrap, cost/borrow/threshold sweeps, the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`stale_nav/`](stale_nav/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
