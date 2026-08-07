# Study 818 — Trend Factor 🌊

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a data-weighted blend of 7 moving-average horizons forecast the cross-section (Han-Zhou-Zhu)? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The famous trend factor **does not replicate** on 50 liquid US mega-caps. The specified long-high-trend / short-low-trend spread is **+1.42 bps/day** (Newey-West *t* = **+0.99**) — the right sign but statistically indistinguishable from zero, only ~**1.45 sd** into a 1,000-permutation placebo (p = 0.066) and weak-and-fading across both eras (*t* = +0.96 / +0.57). Worse for the claim, the blended factor is the **weakest of the three** sorts it is said to *beat*: single-MA(200) timing (*t* = +1.24) and 12-1 momentum (*t* = +1.40) both edge it out. A 20-seed synthetic control recovers a *planted* trend relation cleanly (*t* = +10.28, fires on **1/20** nulls) — so this is a true null, not machinery. *Survivorship: current-membership mega-caps — magnitudes are an upper bound.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The **+1.42 bps/day** gross edge is *already smaller* than the **2.14 bps/day** round-trip friction at a mere 1 bp one-way: the book is net **−0.72 bps/day** even under featherweight costs and bleeds **−22%/yr** at 5 bps. Nothing to trade. |

> **In one sentence:** the celebrated trend factor — a Fama-MacBeth-weighted blend of seven
> moving-average horizons — **adds nothing on liquid US mega-caps**; its long-short spread is a
> flat, insignificant +1.42 bps/day (NW *t* = +0.99), it is *out-performed by the plain single-MA
> and momentum sorts it claims to beat*, and it dies under the lightest costs, so the honest read
> is **claimed signal absent, paycheck a mirage**.

## What we tested

Han, Zhou & Zhu (2016), **"A Trend Factor: Any Economic Gains from Using Information over
Investment Horizons?"**: for each name build normalized moving-average signals `A_L = MA_L(price)
/ price` for `L ∈ {3,5,10,20,50,100,200}`; each period run a cross-sectional (Fama-MacBeth)
regression of the next return on the `A_L` vector, average the *past* slopes, and dot them into
today's signals to get the fitted expected return — the **trend factor** — then sort long-high /
short-low. We take the self-contained daily version on a **liquid 50-name US cross-section
(yfinance daily OHLC, total-return, 2010-01-04 → 2026-06-30)**: the trend factor from a 250-day
rolling slope average, sorted point-in-time (signal known at the close of `t−1`, one shift, zero
look-ahead), with a Newey-West *t*, an explicit **contrast** against single-MA(200) timing and
12-1 momentum, a 1,000-permutation placebo, a two-era robustness cut, a costed long-short timer,
and a 20-seed synthetic positive control. The universe is a **current-membership** survivor set
(`quantlab.universe` opt-in guard) — named on the **Signal** axis. **Dedup:**
[110-faber-timing](../110-faber-timing/) is a **single**-MA timing rule (one horizon, in/out of
market), [438-triple-ma-crossover](../438-triple-ma-crossover/) a fixed **three-MA crossover**,
[518-tsmom](../518-tsmom/) **time-series** (own-return-sign) momentum, and
[507-momentum](../507-momentum/) the plain **12-1 cross-sectional** momentum — none build the
**rolling-Fama-MacBeth-weighted blend of seven MA horizons** this study sorts on (and which,
here, *loses* to the simpler siblings). As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why blending many moving-average horizons *should* forecast better — and why on mega-caps it forecasts nothing (and loses to a plain 200-day rule) |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the spread Newey-West *t*, the single-MA & momentum contrast, the 1,000-permutation placebo, the two-era cut, the cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`trend_factor/`](trend_factor/). Cross-section pulled through the
`quantlab.universe` survivorship guard (current membership → magnitudes are an upper bound).
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
