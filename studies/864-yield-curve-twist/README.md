# Study 864 — Yield-Curve Twist (Butterfly) 🔀

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the butterfly (curve *curvature*) predict forward returns? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The butterfly `2·y10 − y5 − y30` does load on forward **belly-Treasury (IEF)** returns with the right sign and a full-sample Newey-West *t* just past 2 (**+2.34** at 21d, β = **+16.9 bps/1σ**, R² 1.6%), and it is **not** just the 2s10s slope repackaged (the slope control is insignificant, *t* = +0.25). But it **fails the era-stability bar**: the whole effect is a **2002-2009** phenomenon (*t* = +3.32, R² 10%) that is **dead** in 2010-2017 (*t* = −0.18) and 2018-2026 (*t* = +0.59). The quintile version is sub-2 (*t* = 1.66), the incremental *t* drops to 1.98, the *change* (`dfly`, the twist) carries nothing (*t* ≈ 0), and a synthetic null shows the HAC *t* is inflated (real ≈ 1.8 effective σ). Right direction, unstable magnitude. |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | A curvature timer (own IEF on high-fly days, cash otherwise) **loses to buy-and-hold on mean return** at every cost (−0.39 bps/day at 1 bp, −0.61 at 5 bps); its sliver of a Sharpe edge (0.557 vs 0.522) is a cash-parking **volatility artefact** that inverts under realistic friction. |
| **Twist beyond level+slope?** | ![Mixed](https://img.shields.io/badge/Distinct-Mixed-8b949e?style=flat-square) | The *standing curvature* holds marginal (unstable) information the slope does not; the *change* in curvature carries **none**. |

> **In one sentence:** the curve's third factor — its **curvature / butterfly** — does nudge
> forward belly-Treasury returns the right way (cheap belly → belly rallies, NW *t* = +2.34), but
> the entire signal is a pre-2010 relic that vanishes in every modern era and no version pays after
> costs, so the honest read is **a real-but-fragile shape effect, a paycheck mirage**.

## What we tested

Beyond **level** and **slope**, the third principal component of the Treasury curve (Litterman &
Scheinkman 1991) is its **curvature** — the *butterfly* `fly = 2·y10 − y5 − y30` (the belly vs the
wings), and a *twist* is a change in it (`dfly`). A positive fly = a **cheap belly** (10y yield high
relative to the 5y/30y wings); the folklore says it mean-reverts, so it should precede belly
(10-year) bonds outperforming. We build the fly from the CBOE yield indices **^FVX / ^TNX / ^TYX**
and test whether it (and its change) predicts forward **IEF / TLT / SPY** returns
(**2002-07-31 → 2026-06-30**, yfinance daily total-return): a lagged, z-scored predictive
regression with a Newey-West *t*, an **incremental** fit that partials out the 5s10s slope and the
level, a Q5−Q1 quintile sort, a three-era robustness cut, a 500-permutation placebo, a costed
timing overlay, and a 20-seed synthetic positive control. No cross-section, so **no survivorship
bias** (named on the Signal axis for completeness); one documented execution lag (signal at close
`t−1` → return from close `t`); as-of **2026-06-30** (no partial months). **Dedup:**
[66-inverted-yield-curve](../66-inverted-yield-curve/) is the **inversion** (level/slope) recession
bellwether; [132-yield-curve-steepener](../132-yield-curve-steepener/) times on the **2s10s slope**
(here a *control* the butterfly beats out but is tested against); [380-treasury-roll-down](../380-treasury-roll-down/)
is **roll-down / carry** (a level-of-yield mechanism); [581-term-premium](../581-term-premium/) is
the **term premium** (duration compensation along the level dimension). None sorts on the
**curvature / butterfly** — this study's own axis.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a butterfly is, why a cheap belly *should* mean-revert — and why the effect turns out to be a pre-2010 relic |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the HAC regression *t*, the incremental slope-control dedup, the Q5−Q1 spread, the three-era cut, the placebo, the cost math, and the synthetic control (with the HAC over-rejection caveat) |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`curve_twist/`](curve_twist/). Real tape pulled from yfinance (`^FVX`/`^TNX`/`^TYX` +
IEF/TLT/SPY) into this study's own `_cache/`. **Not investment advice** — research & education.
See [LICENSE](../../LICENSE).*
