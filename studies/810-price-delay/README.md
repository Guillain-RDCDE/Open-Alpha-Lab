# Study 810 — Price Delay ⏳

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do slow-to-price (high-delay) names earn a premium (Hou-Moskowitz)? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The famous price-delay premium **fails to replicate** on 50 liquid US mega-caps. The specified long-high-delay / short-low-delay spread is **+2.45 bps/week** (Newey-West *t* = **+0.41**) — the *right sign* (slow names did edge out prompt names) but **statistically indistinguishable from zero**: it sits at placebo p ≈ **0.30**, is flat in both eras (*t* = +0.17 / +0.39), and a 20-seed synthetic control recovers a *planted* delay premium cleanly (*t* = +10.83, fires on ~**1/20** nulls, the nominal 5%). The premium is a **small / illiquid / low-coverage-stock** effect; mega-caps — priced to the millisecond — are exactly where it should not appear. *Survivorship: current-membership mega-caps — magnitudes are an upper bound.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The +2.45 bps/week gross edge is **smaller than one weekly round-trip** at a mere 1 bp one-way (**2.96 bps/week** friction), so the book loses money net from the first basis point: **−0.51 bps/week** at 1 bp, −8.51 at 5 bps. |

> **In one sentence:** the celebrated price-delay premium — slow-to-price names should
> out-earn — **does not survive on liquid US mega-caps**; here it is the right sign but a
> coin-flip (NW *t* = +0.41, placebo p ≈ 0.30), and the tiny gross edge is eaten by the very
> first basis point of cost, so the honest read is **claimed signal absent, paycheck a mirage**.

## What we tested

Hou & Moskowitz (2005), **"Market Frictions, Price Delay, and the Cross-Section of Expected
Returns"**: a stock into which market information diffuses **slowly** should command a **return
premium**. Their **delay** measure regresses a name's weekly return on the contemporaneous
market plus lags and reads off how much explanatory power the *lagged* terms add
(`delay = 1 − R²_contemp-only / R²_with-lags`). We take the self-contained weekly version on a
**liquid 50-name US cross-section (yfinance daily OHLC, total-return, 2010-01-04 → 2026-06-30,
resampled to W-FRI weekly)**: each name's **trailing-52-week delay** (contemporaneous market —
the equal-weight cross-sectional mean — plus **4 weekly lags**), sorted point-in-time (signal
known at the close of `t−1`, one shift, zero look-ahead) **long top-30% / short bottom-30%**,
with a Newey-West *t* on the weekly spread, a 1,000-permutation placebo, a two-era robustness
cut, a costed long-short timer, and a 20-seed synthetic positive control. The universe is a
**current-membership** survivor set (`quantlab.universe` opt-in guard) — named on the **Signal**
axis. **Dedup:** [140-amihud-illiquidity](../140-amihud-illiquidity/) is a volume-scaled
*price-impact* level, not a diffusion-speed R² ratio; [379-etf-lead-lag](../379-etf-lead-lag/)
*trades* the ETF→member lagged move rather than sorting on a name's lagged-market loading;
[512-high-volume-premium](../512-high-volume-premium/) is the opposite pole — a high-*visibility*
volume shock, not slow-to-price neglect. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why slow-to-price names *should* earn a premium — and why on mega-caps there is nothing to sort on |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the spread Newey-West *t*, the pooled Welch book test, the 1,000-permutation placebo, the two-era cut, the cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`price_delay/`](price_delay/). Cross-section pulled through the `quantlab.universe`
survivorship guard (current membership → magnitudes are an upper bound). **Not investment
advice** — research & education. See [LICENSE](../../LICENSE).*
