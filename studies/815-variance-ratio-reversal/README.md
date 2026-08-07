# Study 815 — Variance-Ratio Reversal 🔬📏

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do low-VR (mean-reverting) names offer a tradable reversal (Lo & MacKinlay)? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The variance-ratio reversal **fails to replicate — and inverts** — on 50 liquid US mega-caps. The specified long-low-VR / short-high-VR spread is **−2.69 bps/day** (Newey-West *t* = **−2.44**): the **high**-VR (trending) names actually *out-earned* the mean-reverters (2010–2026). It is significant only at the headline 120-day window and *opposite in sign* to the claim, and it is **fragile** — insignificant in 2018–2026 (*t* = −1.36) and gone at 63-day / 252-day windows (*t* = −0.66 / −0.53). The observed value sits ≈**2.9σ into the left tail** of a 1,000-permutation placebo, and a 20-seed synthetic control recovers a *planted* low-VR premium cleanly (*t* = +9.75, fires on **1/20** nulls) — so the engine is sound and there is simply no mean-reversion premium to harvest. *Survivorship: current-membership mega-caps — magnitudes are an upper bound.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The specified book loses money gross and net (**−4.83 bps/day** at 1 bp one-way, −12.83 at 5 bps). Even the data-mined *sign-flip* (long high-VR) earns only +2.69 bps/day gross, which the **2.14 bps/day** round-trip friction at a mere 1 bp already eats — a Mirage in either direction. |

> **In one sentence:** the celebrated Lo-MacKinlay variance-ratio reversal — mean-reverting
> (low-VR) names should out-earn — **does not survive on liquid US mega-caps**; here the
> relation is significant but *reversed* and fragile (the trending names out-earned, NW *t* =
> −2.44), and no version of the book survives costs, so the honest read is **claimed signal
> absent, paycheck a mirage**.

## What we tested

Lo & MacKinlay (1988), **"Stock Market Prices Do Not Follow Random Walks"**: the **variance
ratio** `VR(q) = Var(q-day return) / (q × Var(1-day return))` diagnoses a random-walk departure —
`VR < 1` mean-reverting, `VR > 1` trending — so a long low-VR / short high-VR book is meant to
harvest a **reversal** premium. We take the self-contained daily version on a **liquid 50-name US
cross-section (yfinance daily OHLC, total-return, 2010-01-04 → 2026-06-30)**: each name's
**trailing-120-day Lo-MacKinlay `VR(q=5)`** (overlapping, bias-corrected, vectorised), sorted
point-in-time (signal known at the close of `t−1`, one shift, zero look-ahead), with a Newey-West
*t* on the daily spread, a 1,000-permutation placebo, two-era & two-window robustness cuts, a
costed long-short timer, and a 20-seed synthetic positive control. The universe is a
**current-membership** survivor set (`quantlab.universe` opt-in guard) — named on the **Signal**
axis. **Dedup:** [397-hurst-regime](../397-hurst-regime/) uses the multi-scale **Hurst exponent**,
not the single-horizon variance ratio; [398-entropy-efficiency](../398-entropy-efficiency/) uses
**permutation entropy** (ordinal information), not a second-moment ratio;
[329-one-month-reversal](../329-one-month-reversal/) sorts on the **level of the last month's
return**, not on the return's **autocorrelation shape**. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a variance ratio below 1 means, why mean-reverters *should* pay — and why on mega-caps the trenders out-earned |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the Lo-MacKinlay VR(5) signal, the spread Newey-West *t*, the pooled Welch book test, the 1,000-permutation placebo, the two-era & two-window cuts, the cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`variance_ratio/`](variance_ratio/). Cross-section pulled through the
`quantlab.universe` survivorship guard (current membership → magnitudes are an upper bound).
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
