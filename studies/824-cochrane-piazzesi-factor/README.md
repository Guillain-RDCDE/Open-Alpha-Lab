# Study 824 — Cochrane-Piazzesi Factor 🧮📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does one tent of forwards forecast bond excess returns (Cochrane-Piazzesi)? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The regression *does* produce an in-sample R² of **0.226** with correctly **tent-shaped** loadings (peak on the 5→10y forward) and a strong first era (NW *t* = **+3.90**) — the claim's fingerprint is there. But it is **not a robust edge**: a block placebo puts that R² at **p = 0.21** (persistent regressors alone give R² ≈ **0.18**), the **out-of-sample R² is −0.27** (worse than a constant), the second era is insignificant (*t* = +1.72), and the headline HAC *t* = **+2.62** sits *inside* the synthetic null's own distribution of that size-distorted statistic (mean **+2.08**, fires 11/20 nulls). Right shape, no robust forecasting power — the Bauer-Hamilton spurious-regression reading. *Signal caveat: a coarse 0.25/5/10/30y forward grid, a proxy for CP's Fama-Bliss 1..5y zeros.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | A duration timer that owns TLT when the *out-of-sample* CP forecast is rich earns a net Sharpe of ~**0.02** (2 bps) / **0.005** (5 bps) versus **0.22** for simply holding TLT — the signal *subtracts* value. No paycheck. |

> **In one sentence:** the celebrated Cochrane-Piazzesi return-forecasting factor shows the
> right tent-shaped loadings and a fat in-sample R² on this coarse curve, but it is **≈0.9σ from a
> spurious persistent-regressor fit** (placebo p = 0.21), goes **negative out of sample**, and its
> timed book earns nothing — an in-sample mirage, not a robust forecaster.

## What we tested

Cochrane & Piazzesi (2005), **"Bond Risk Premia"**: a *single* tent-shaped linear combination of
**forward rates** forecasts the one-year-ahead **excess return** of Treasury bonds across all
maturities, with an R² a plain curve slope cannot match. We rebuild the factor from the coarse
constant-maturity yields yfinance exposes (**`^IRX` 0.25y, `^FVX` 5y, `^TNX` 10y, `^TYX` 30y**,
2002-01-02 → 2026-06-30): build the four implied **forwards**, regress the **average 252-day
excess return** of the `SHY / IEF / TLT` bond ETFs (over the `^IRX` risk-free) on the forward
vector, and read the fitted **CP factor**, its R², and a Newey-West *t* with lags scaled to the
252-day overlap. Because near-unit-root yields make even the HAC *t* size-distorted (our synthetic
null fires it 11/20), we grade on a **Campbell-Thompson out-of-sample R²** and a **block-rotation
placebo**, plus a two-era cut, a costed duration timer, and a 20-seed synthetic control. One
documented execution lag (signal at close `t−1`, held `t`); as-of **2026-06-30**. The coarse
forward grid is a **proxy** for CP's clean Fama-Bliss zeros — named on the **Signal** axis.
**Dedup:** [581-term-premium](../581-term-premium/) times TLT with a *single* term-premium proxy,
not the whole forward vector collapsed to one factor; [132-yield-curve-steepener](../132-yield-curve-steepener/)
trades the raw 10y−3m **slope**; [66-inverted](../66-inverted/) is the curve **inversion** regime bit;
[380-curve-roll-down](../380-curve-roll-down/) is deterministic **roll/carry**, not a risk-premium forecaster.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a "return-forecasting factor" is, why the tent is the right shape — and why a fat R² on persistent yields is almost free |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the HAC predictive regression, the size-distorted *t*, the out-of-sample R², the block placebo, the two-era cut, the costed timer, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`cp_factor/`](cp_factor/). Yields + bond ETFs pulled once via yfinance into this study's own `_cache/` (coarse constant-maturity grid → a proxy for CP's Fama-Bliss zeros, named on the Signal axis). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
