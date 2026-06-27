# Study 537 -- Factor-Momentum

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

> Don't ask whether *stocks* trend -- ask whether the **factors** do. Factors that did well
> recently keep doing well, so time the factors on their own past returns and pocket the
> meta-premium.

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Timed meta-premium **+2.10%/yr** (best 3m lookback), one-sample *t* = **+0.75**, HAC *t* = +1.06, placebo **p = 0.096** (seed-robust) -- below the |*t*| >= 2 bar. Weak (not None) because the *premise* holds: factors are persistent and timing beats holding them static. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Net of 10 bps x turnover + borrow, the best config nets **+1.16%/yr** (net *t* = 0.41); 1m/12m go net-negative. Sharpe <= 0.16, max DD -30% to -55%, and the premium leans on a costly short leg. |
| **Do the factors trend?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | Lag-1 factor autocorrelation is positive and consistent (**~0.15**, four of five factors). The Ehsani-Linnainmaa premise survives even where the tradable meta-premium does not. |

> **In one sentence:** factors really do trend -- their returns autocorrelate (~0.15) and timing
> them turns a -3.36%/yr static loss into a +2.10%/yr gain in the Ehsani-Linnainmaa direction --
> but on a 22-year, 5-factor large-cap survivor panel the timed meta-premium is too thin to clear
> *t* = 2 or survive costs: Weak signal, Mirage tradability, premise Confirmed.

## What we tested

Ehsani & Linnainmaa (2022): the factors themselves exhibit time-series momentum. We build five
**price-derived** long-short factors -- 12-1 momentum, low-vol, low-beta, short-reversal, and a
size proxy -- from a fixed 40-name large-cap survivor basket (yfinance daily closes,
2004-08 -> 2026-05, 262 monthly observations; the in-progress June 2026 month is excluded --
no partial periods). Each month we hold a factor **long** if its
trailing return is positive and **short** if negative (one-month execution lag; the factor panel
is already lagged vs its signal), and average across factors. We test the timed mean against zero
(one-sample + HAC *t*), against a **time-shuffle placebo** that destroys factor autocorrelation,
and net of one-way costs x turnover + borrow on short sleeves. A deterministic synthetic control
plants a known AR(1) persistence to prove the engine recovers real trends and refuses to invent
them from noise. The basket is **survivorship-biased** (long-leg tilt) and **price-only** (no
honest point-in-time fundamentals for a value/quality factor) -- both named.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | factors-as-assets in plain language, the "do factors trend?" picture, timing-vs-holding, synthetic control, honest verdict |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | autocorrelation of factors, lookback sweep, time-shuffle placebo, per-factor timed attribution, costs x turnover, faithful-engine power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`factor_momentum/`](factor_momentum/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
