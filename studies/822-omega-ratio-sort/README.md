# Study 822 — Omega-Ratio Sort ⚖️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a full gain/loss ratio (Omega) beat plain trailing Sharpe? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The Keating-Shadwick Omega advantage **does not materialise** on 50 liquid US mega-caps. The specified long-high-Omega / short-low-Omega spread is **+1.19 bps/day** (Newey-West *t* = **+0.76**) — right sign, but nowhere near the |*t*| ≥ 2 bar, flat in both eras (*t* = +0.16 / +0.80) and inside a 1,000-permutation placebo (p = 0.12). Crucially the Omega sort is **+0.996 rank-identical to a plain trailing-Sharpe sort** and earns *less* than it (Sharpe +1.29 bps) — the "whole distribution" Omega is sold on adds nothing over mean/vol. A 20-seed synthetic control recovers a *planted* effect cleanly (fires on **1/20** nulls ≈ the nominal 5%), so this is a true null. *Survivorship: current-membership mega-caps — magnitudes are an upper bound.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The book is **net-negative at every realistic cost** (**−0.95 bps/day** at 1 bp one-way, −8.95 at 5 bps): the thin +1.19 bps gross edge is already thinner than the 2.14 bps/day round-trip friction. |

> **In one sentence:** the celebrated Omega ratio — a "universal" gain/loss measure that
> reads the whole return distribution — **is a near-perfect twin of the Sharpe ratio on
> daily equities** (rank corr +0.996), so its cross-sectional sort inherits Sharpe's
> insignificance rather than curing it, and no version survives costs — **claimed edge over
> Sharpe absent, paycheck a mirage**.

## What we tested

Keating & Shadwick (2002), **"A Universal Performance Measure"**: the **Omega ratio** at
threshold 0, `Ω(0) = E[max(r,0)] / E[max(−r,0)]`, is a gain/loss ratio that uses *every*
moment of the return distribution, so a long-high-Omega / short-low-Omega sort should **beat
a plain trailing-Sharpe sort**. We take the self-contained daily version on a **liquid 50-name
US cross-section (yfinance daily OHLC, total-return, 2010-01-04 → 2026-06-30)**: each name's
**trailing 12-1 Omega(0)** (average gain ÷ average loss over a 252-day window ending ~1 month
ago), sorted point-in-time (signal known at the close of `t−1`, one further shift, zero
look-ahead), with a Newey-West *t* on the daily spread, a head-to-head against the identical
**Sharpe** and **low-vol** sorts (with per-day rank overlaps), a 1,000-permutation placebo, a
two-era robustness cut, a costed long-short timer, and a 20-seed synthetic positive control.
The universe is a **current-membership** survivor set (`quantlab.universe` opt-in guard) —
named on the **Signal** axis. **Dedup:** [814-trailing-sharpe-anomaly](../814-trailing-sharpe-anomaly/)
is the **direct comparator** (the Sharpe sort Omega is measured against — and is +0.996
rank-identical to); [330-low-volatility-anomaly](../330-low-volatility-anomaly/) is the raw
**low-vol** tilt (Omega ~ (−vol) is only +0.075 here, so low-vol is *not* the driver). As-of
**2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what Omega adds over Sharpe in principle — and why on real returns the two sorts turn out to be the same sort |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the spread Newey-West *t*, the Omega-vs-Sharpe-vs-low-vol head-to-head with rank overlaps, the 1,000-permutation placebo, the two-era cut, the cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`omega_ratio/`](omega_ratio/). Cross-section pulled through the
`quantlab.universe` survivorship guard (current membership → magnitudes are an upper bound).
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
