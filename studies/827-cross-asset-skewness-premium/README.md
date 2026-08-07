# Study 827 — Cross-Asset Skewness Premium 🎲🌐

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do low-skew *asset classes* out-earn high-skew ones (the asset-class analogue of Study 803)? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The single-name realized-skewness reversal **does not carry up to the asset-class level.** Across nine class-proxy ETFs the long-low-skew / short-high-skew spread is **+13.73 bps/month** (Newey-West *t* = **+0.62**): the *sign* matches the claim (low-skew classes edged out high-skew ones) but the magnitude is **statistically zero** — only ≈**0.83σ** into the right tail of a 1,000-permutation placebo (p = **0.20**), **flipping sign across the two eras** (−15.21 bps early / +39.54 late) and insignificant at every lookback (63/126/252-day *t* = +1.04 / +0.62 / +0.25). A 20-seed synthetic control fires on a *planted* relation (*t* = +2.52) and stays silent on the null (**2/20**), so this is an honest null, not a broken sort — with only nine classes there is too little skew dispersion for the effect to exist. *Survivorship: fixed current-membership class-proxy ETFs — milder than a single-name universe.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The gross spread is already insignificant (annualised Sharpe **+0.15**); a token **5 bps** one-way cost erases it to zero (**+7.57 bps/mo** net at 1 bp, *t* = +0.35; **−0.43 bps/mo** at 5 bps). Nothing survives friction because there was nothing there to begin with. |

> **In one sentence:** the celebrated "lottery names under-earn" reversal — robust across single
> **stocks** — **does not exist across asset classes**: sorting nine class ETFs on their realized
> skew gives a right-signed but statistically zero spread (NW *t* = +0.62) that flips sign era to
> era and dies at the first basis point of cost, so the honest read is **claimed signal absent,
> paycheck a mirage**.

## What we tested

The asset-class analogue of the single-name realized-skewness reversal (Amaya, Christoffersen,
Jacobs & Vasquez 2015; Study 803): sort a cross-section of asset classes on their recent **realized
return skewness**; if lottery-overpricing operates at the class level, low-skew classes should
out-earn, so a long-low-skew / short-high-skew book earns a *positive* spread. We take **nine
liquid asset-class ETFs** (SPY, EFA, EEM, TLT, LQD, HYG, GLD, DBC, VNQ — yfinance daily total-return
closes, 2007-01-03 → 2026-06-30): each class's **trailing-126-day realized skewness** of daily
returns (vectorised third moment), sorted **monthly** point-in-time (signal at month-end `m−1`, hold
month `m`, one shift, zero look-ahead), with a Newey-West *t* on the monthly spread, a
1,000-permutation asset-label placebo, a two-era and multi-window robustness cut, a costed long-short
timer, and a 20-seed synthetic positive control. The universe is a small **current-membership**
class-proxy set — named on the **Signal** axis. **Dedup:** [803-realized-skewness-reversal](../803-realized-skewness-reversal/)
is the **single-name** version (a stock cross-section, not asset classes); [660-carry-everywhere](../660-carry-everywhere/)
runs cross-asset **carry** (the yield signal, not the third moment); [638-value-momentum-everywhere](../638-value-momentum-everywhere/)
runs cross-asset **value & momentum** (level & trend, not skewness). As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why lottery-like classes *should* under-earn — and why with only nine classes the effect has nothing to bite on |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the spread Newey-West *t*, the pooled Welch book test, the 1,000-permutation placebo, the two-era & window cut, the cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`cross_asset_skew/`](cross_asset_skew/). Nine asset-class ETFs pulled via yfinance
(total-return closes) into this study's own `_cache/`. **Not investment advice** — research &
education. See [LICENSE](../../LICENSE).*
