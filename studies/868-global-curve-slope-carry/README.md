# Study 868 — Global Curve-Slope Carry 🌍

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a high-carry / steep-curve cross-sectional sort pay a duration holder (Koijen-Moskowitz-Pedersen-Vrugt carry)? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | On six US + international sovereign-bond ETFs (2007-2026) the primary **yield-to-duration** carry sort earns the **wrong sign**: **−20.17 bps/mo**, Newey-West *t* = **−1.45**, and a column-permutation placebo puts the observed *deep in the left tail* (*p* = **0.94** — a random leg assignment beats it 94 % of the time); its "low-carry" short leg (+30.16 bps) actually *out-earns* its "high-carry" long leg (+9.99 bps). The plainer **raw realized-yield** sort is **dead flat** (**+3.16 bps/mo**, NW *t* = **+0.20**, placebo *p* = 0.44). Both flip sign across eras (the only \|*t*\| ≥ 2 reading is the wrong-signed 2016-2021 book, −40.24 bps, *t* = −3.02) and across formation windows. A 20-seed synthetic control recovers a *planted* carry cleanly (mean *t* = **+18.17**, fires **2/20** on the null) — so the null is real, not a broken engine. *Survivorship (currently-listed funds only), a short full cross-section (BNDX from 2013), and a price-only carry proxy are all named on the Signal axis.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Every variant loses money once costed (yield-to-duration net **−28.24 bps/mo** at 5 bps, raw carry net **−4.81 bps/mo** at 5 bps; net Sharpe < 0 in all cases) on ~0.35 monthly turnover plus short borrow. No costed net edge survives. |

> **In one sentence:** ranking US + international sovereign-bond ETFs by a curve-steepness /
> carry proxy and going long the high-carry, short the flat markets **does not work** — the
> yield-to-duration sort is actually *negative* (NW *t* = −1.45, beaten by random assignment
> 94 % of the time) and the raw-carry sort is dead flat (NW *t* = +0.20), both flip sign
> across eras, and everything loses money after costs: **claimed signal absent, paycheck a mirage**.

## What we tested

Koijen, Moskowitz, Pedersen & Vrugt (2018), **"Carry"**, applied to the sovereign-bond
curve: a **steep** yield curve pays a duration holder (yield + roll-down), so ranking bond
markets by a curve-steepness / carry proxy and going **long the high-carry, short the flat**
should pay. We take six liquid **US + international sovereign-bond ETFs** (yfinance
total-return, month-end, 2007-2026) — `SHY`, `IEF`, `TLT` (US curve) plus `BWX`, `IGOV`,
`BNDX` (international) — score each by a **yield-to-duration carry proxy** (a trailing
36-month realized yield ÷ published effective duration) known at the close of `t−1` (one
shift, zero look-ahead), and cost the dollar-neutral high-minus-low book with a Newey-West
*t*, a 3,000-draw column-permutation placebo, a three-era cut, a 24/36/48/60-month window
sweep, and a 20-seed synthetic positive control, always **benchmarked against naive
equal-weight buy-and-hold**. Survivorship (currently-listed funds only), the short full
cross-section (`BNDX` lists from 2013), and the price-only carry proxy are named on the
**Signal** axis. **Dedup:**
[829-global-sovereign-bond-momentum](../829-global-sovereign-bond-momentum/) is a
**time-series trend** signal, not a cross-sectional carry sort;
[826-treasury-duration-bab](../826-treasury-duration-bab/) is a **US-only** beta-neutral
BAB book, not a US+international carry sort; [380-curve-roll-down](../380-curve-roll-down/)
is a **single-curve** roll-down timer, not the cross-market sort; and
[660-carry-everywhere](../660-carry-everywhere/) is the **cross-asset-class** carry factor,
not the isolated sovereign-bond-curve sleeve. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a steep curve *should* pay a duration holder — and why on these ETFs the carry sort points the wrong way |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the Newey-West *t*, the buy-and-hold benchmark, the column-permutation placebo, the three-era cut, the window sweep, the cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`curve_slope_carry/`](curve_slope_carry/). Six sovereign-bond-ETF total-return
closes pulled via yfinance into this study's own `_cache/`; the reproducible core runs
offline. Currently-listed funds only, and a price-only carry proxy → magnitudes are an
upper bound with limited power. **Not investment advice** — research & education. See
[LICENSE](../../LICENSE).*
