# Study 821 — Turnover Volatility 🌀

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do erratic-turnover names go on to earn *less* (Chordia-Subrahmanyam-Anshuman)? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The famous turnover-**variability** premium **fails to replicate** on 50 liquid US mega-caps. The specified long-low-vol / short-high-vol spread is a statistically **insignificant −1.70 bps/day** (Newey-West *t* = **−1.73**, \|t\| < 2), if anything faintly *wrong-signed* (the erratic names slightly *out*-earned), carried entirely by the pre-2018 era (−3.00 bps, *t* = −2.21) and gone thereafter (−0.51 bps, *t* = −0.36). The dollar-volume variant agrees (−1.64 bps, *t* = −1.68), the placebo shows no reliable spread, and a 20-seed synthetic control recovers a *planted* CSA relation cleanly (*t* = +9.33, fires on **0/20** nulls) — so the machinery is sound; the claimed edge is simply **absent** on mega-caps (a small/illiquid-stock phenomenon). *Survivorship: current-membership mega-caps — magnitudes are an upper bound.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The specified book loses money gross and net (**−3.84 bps/day** at 1 bp one-way, −11.84 at 5 bps). Even a data-mined *sign-flip* (long high-vol) earns only +1.70 bps/day gross, which the **2.14 bps/day** round-trip friction at a mere 1 bp already eats — a Mirage in either direction. |

> **In one sentence:** the celebrated coefficient-of-variation-of-turnover premium — erratic-turnover
> names should under-earn — **does not survive on liquid US mega-caps**; here the spread is an
> insignificant −1.70 bps/day (NW *t* = −1.73) that is if anything faintly reversed and dies pre-2018,
> and no version of the book survives costs, so the honest read is **claimed signal absent, paycheck a mirage**.

## What we tested

Chordia, Subrahmanyam & Anshuman (2001), **"Trading activity and expected stock returns"**: beyond the
*level* of trading, its **variability** predicts returns negatively — sort stocks on the **coefficient of
variation of daily turnover** (std/mean over a trailing window); the erratic-turnover names should
under-earn, so a long low-vol / short high-vol book should earn a *positive* spread. We take the
self-contained daily version on a **liquid 50-name US cross-section (yfinance daily OHLC + Volume,
total-return, 2010-01-04 → 2026-06-30)**: each name's **trailing-63-day CV of daily share turnover**
(scale-invariant, so raw Volume ≡ fixed-shares turnover), sorted point-in-time (signal known at the close
of `t−1`, one shift, zero look-ahead), with a Newey-West *t* on the daily spread, a 1,000-permutation
placebo, a two-era robustness cut, a dollar-volume variant, a costed long-short timer, and a 20-seed
synthetic positive control. The universe is a **current-membership** survivor set (`quantlab.universe`
opt-in guard) — named on the **Signal** axis. **Dedup:** [141-turnover-anomaly](../141-turnover-anomaly/)
tests the **level** of turnover (Datar-Naik-Radcliffe), not its variability;
[140-amihud-illiquidity](../140-amihud-illiquidity/) tests Amihud's price-**impact** illiquidity, not the
dispersion of trading activity; [512-high-volume-premium](../512-high-volume-premium/) tests the kick after
an **abnormal single-day volume spike**, not the trailing CV of turnover. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why *erratic* liquidity *should* command a discount — and why on mega-caps nothing reliable showed up |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the spread Newey-West *t*, the pooled Welch book test, the 1,000-permutation placebo, the two-era cut, the dollar-volume variant, the cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`turnover_vol/`](turnover_vol/). Cross-section pulled through the
`quantlab.universe` survivorship guard (current membership → magnitudes are an upper bound).
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
