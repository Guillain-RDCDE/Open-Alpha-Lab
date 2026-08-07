# Study 829 — Global Sovereign-Bond Momentum 🌍📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a bond market's own 12-1 trend predict its next month (Moskowitz-Ooi-Pedersen)? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The specified **long-short** 12-1 time-series-momentum book is **flat** on five global sovereign-bond ETFs (2007-2026): **+0.26 bps/mo**, Newey-West *t* = **+0.02**, block-rotation placebo *p* = 0.72. The **long-only** variant *looks* positive (+12.27 bps/mo) but that is **bond beta, not trend**: it under-earns naive equal-weight buy-and-hold (**+24.56 bps/mo**), its placebo observation sits *below* the rotation null (*p* = 0.53), its NW *t* = **+1.75** misses the \|*t*\| ≥ 2 bar, it fades era-by-era (2020-2026: +2.25 bps, *t* = +0.29), and it is fragile across lookbacks (*t* ∈ [1.2, 1.8]). A 20-seed synthetic control recovers a *planted* trend cleanly (mean *t* = +13.4, fires **0/20** on the null) — so the null is real, not a broken engine. *Survivorship: currently-listed funds only, and a 5-ETF panel — an upper bound with limited power, named on the Signal axis.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The long-short book loses money once costed (**−3.18 bps/mo** at 5 bps one-way, −4.48 at 10 bps; net Sharpe < 0). The long-only book's costed net Sharpe (~0.29) is **diluted, costed bond beta** that under-earns buy-and-hold — no costed net edge survives. |

> **In one sentence:** time-series momentum on global government-bond ETFs — long the up-trend,
> short the down-trend — **does not work**: the long-short book is dead flat (NW *t* = 0.02), the
> long-only version is just bond beta that under-earns buy-and-hold, and nothing survives costs, so
> the honest read is **claimed signal absent, paycheck a mirage**.

## What we tested

Moskowitz, Ooi & Pedersen (2012), **"Time Series Momentum"**, applied to foreign / global
government bonds: measure each market's **12-minus-1-month total-return trend** and go **long the
positive-trend, short the negative-trend** markets (or long-only trend timing). We take five
liquid **global sovereign-bond ETFs (yfinance total-return, month-end, 2007-01-31 → 2026-06-30)** —
`BWX`, `IGOV` (ex-US developed), `BNDX` (USD-hedged global), `EMB` (EM USD), `IEF` (US anchor) —
sign each by its 12-1 momentum known at the close of `t−1` (one shift, zero look-ahead), and score
the equal-weight strategy with a Newey-West *t*, a 3,000-draw block-rotation placebo, a three-era
robustness cut, a lookback sweep, a costed backtest, and a 20-seed synthetic positive control,
always **benchmarked against naive buy-and-hold**. Survivorship (currently-listed funds only) and
the small panel's limited power are named on the **Signal** axis. **Dedup:**
[795-corporate-bond-momentum](../795-corporate-bond-momentum/) is a **cross-sectional** sort within
**corporate** bonds, not a **time-series** trend on **sovereigns**;
[518-time-series-momentum](../518-time-series-momentum/) is the **general** cross-asset TSMOM
factor, not the isolated & costed global-sovereign sleeve;
[662-em-local-bonds](../662-em-local-bonds/) is an EM **carry / level** study, not **trend**. As-of
**2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why bond trends *should* persist — and why on these ETFs the "edge" is just owning bonds |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the Newey-West *t*, the buy-and-hold benchmark, the block-rotation placebo, the three-era cut, the lookback sweep, the cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`sovereign_momentum/`](sovereign_momentum/). Real tape via yfinance (total-return
month-end closes), cached under `_cache/`; currently-listed funds only → magnitudes are an upper
bound with limited power. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
