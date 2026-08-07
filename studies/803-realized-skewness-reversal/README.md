# Study 803 — Realized-Skewness Reversal 🎲📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do high-realized-skew names go on to earn *less* (Amaya et al)? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The famous negative skew→return premium **fails to replicate — and inverts** — on 50 liquid US mega-caps. The specified long-low-skew / short-high-skew spread is **−3.19 bps/day** (Newey-West *t* = **−3.04**): the lottery-like **high**-skew names actually *out-earned* the boring low-skew ones (2010–2026). It is significant but *opposite in sign* to the claim, holds in both eras (*t* = −2.12 / −2.27), sits ≈**3.65σ into the left tail** of a 1,000-permutation placebo, and a 20-seed synthetic control recovers a *planted* Amaya relation cleanly (fires on **0/20** nulls) — so the sign-reversal is real, not machinery. The realized-skewness premium is a **small-and-illiquid-stock** phenomenon; mega-caps are exactly where it should not appear. *Survivorship: current-membership mega-caps — magnitudes are an upper bound.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The specified book loses money gross and net (**−5.33 bps/day** at 1 bp one-way, −13.33 at 5 bps). Even the data-mined *sign-flip* (long high-skew) earns only +3.19 bps/day gross, which the **2.14 bps/day** round-trip friction at a mere 1 bp already eats — a Mirage in either direction. |

> **In one sentence:** the celebrated realized-skewness premium — high-skew names should
> under-earn — **does not survive on liquid US mega-caps**; here the relation is significant
> but *reversed* (the lottery names out-earned, NW *t* = −3.04), and no version of the book
> survives costs, so the honest read is **claimed signal absent, paycheck a mirage**.

## What we tested

Amaya, Christoffersen, Jacobs & Vasquez (2015), **"Does Realized Skewness Predict the
Cross-Section of Equity Returns?"**: sort stocks on their recent **realized return skewness**;
the theory says high-skew (lottery-like) names are over-priced and under-earn, so a long
low-skew / short high-skew book should earn a *positive* spread. We take the self-contained
daily version on a **liquid 50-name US cross-section (yfinance daily OHLC, total-return,
2010-01-04 → 2026-06-30)**: each name's **trailing-21-day realized skewness** of daily returns
(vectorised third moment), sorted point-in-time (signal known at the close of `t−1`, one shift,
zero look-ahead), with a Newey-West *t* on the daily spread, a 1,000-permutation placebo, a
two-era robustness cut, a costed long-short timer, and a 20-seed synthetic positive control.
The universe is a **current-membership** survivor set (`quantlab.universe` opt-in guard) —
named on the **Signal** axis. **Dedup:** [503-expected-idiosyncratic-skewness](../503-expected-idiosyncratic-skewness/)
tests the **ex-ante / modelled** skewness (Boyer-Mitton-Vorkink), not the **realized** moment;
[504-coskewness](../504-coskewness/) tests **systematic** co-skewness with the market, not a
name's **own** skew; [365-lottery-max-effect](../365-lottery-max-effect/) uses the single
**MAX** daily return, not the full third moment. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a lottery-like tape *should* signal lower future returns — and why on mega-caps the opposite happened |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the spread Newey-West *t*, the pooled Welch book test, the 1,000-permutation placebo, the two-era cut, the cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`realized_skewness/`](realized_skewness/). Cross-section pulled through the
`quantlab.universe` survivorship guard (current membership → magnitudes are an upper bound).
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
