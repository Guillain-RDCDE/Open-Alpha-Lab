# Study 807 — Salience-Theory Returns ✨📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do salient-*upside* names go on to earn *less* (Cosemans & Frehen)? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The salience-theory premium **does not show up** on 50 liquid US mega-caps. The specified long-low-ST / short-high-ST spread is **−1.00 bps/day** (Newey-West *t* = **−0.78**) — indistinguishable from zero, flat in both eras (*t* = −0.60 / −0.55), only ~**1.1σ** from the centre of a 1,000-permutation placebo (p = 0.87); the low-ST and high-ST books earn essentially the same (+7.49 vs +8.49 bps, Welch *t* = −0.37). A 20-seed synthetic control recovers a *planted* salience relation cleanly (*t* = +4.31, fires on **1/20** nulls), so the flat tape is a real null, not machinery. The effect is documented on the **broad, small-cap-inclusive** cross-section; mega-caps are exactly where it should not appear. *Survivorship: current-membership mega-caps — magnitudes are an upper bound.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | There is no gross edge to harvest (−1.00 bps/day), and the book bleeds net (**−3.14 bps/day** at 1 bp one-way, −11.14 at 5 bps). Even the data-mined *sign-flip* (long high-ST) earns only +1.00 bps/day gross — less than half the **2.14 bps/day** round-trip friction at a mere 1 bp. A Mirage in either direction. |

> **In one sentence:** the celebrated salience-theory premium — salient-upside names should
> under-earn — **leaves no footprint on liquid US mega-caps** (NW *t* = −0.78, flat in both eras,
> right in the placebo noise), and with no gross edge no version of the book survives costs, so the
> honest read is **claimed signal absent, paycheck a mirage**.

## What we tested

Cosemans & Frehen (2021), **"Salience Theory and Stock Prices: Empirical Evidence"** (applying the
Bordalo-Gennaioli-Shleifer salience model): over the trailing month each name's daily salience versus
the market is `σ = |rᵢ−rₘ| / (|rᵢ|+|rₘ|+θ)` (θ=0.1); the days are ranked and given declining decision
weights `δ^rank` (δ=0.7, most-salient over-weighted); the **salience-theory value** ST is the
salience-weighted mean of market-excess returns. The theory says a high-ST (salient-upside) name is
over-priced and under-earns, so a long low-ST / short high-ST book should earn a *positive* spread. We
take the self-contained daily version on a **liquid 50-name US cross-section (yfinance daily OHLC,
total-return, 2010-01-04 → 2026-06-30)**: the trailing-21-day ST (vectorised salience ranking), sorted
point-in-time (signal known at the close of `t−1`, one shift, zero look-ahead), with a Newey-West *t*
on the daily spread, a 1,000-permutation placebo, a two-era robustness cut, a costed long-short timer,
and a 20-seed synthetic positive control. The universe is a **current-membership** survivor set
(`quantlab.universe` opt-in guard) — named on the **Signal** axis. **Dedup:**
[806-prospect-theory-value](../806-prospect-theory-value/) tests the **prospect-theory** value
(S-shaped, probability-weighted valuation of a name's *own* returns), not a market-contrast salience
ranking; [365-lottery-max-effect](../365-lottery-max-effect/) uses the single **MAX** daily return,
not a salience-weighted mean of every day; [503-expected-idiosyncratic-skewness](../503-expected-idiosyncratic-skewness/)
uses a **modelled ex-ante** third moment, not the realised market-relative salience tape. As-of
**2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a salient-*upside* month *should* signal lower future returns — and why on mega-caps nothing shows up |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the spread Newey-West *t*, the pooled Welch book test, the 1,000-permutation placebo, the two-era cut, the cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`salience_theory/`](salience_theory/). Cross-section pulled through the
`quantlab.universe` survivorship guard (current membership → magnitudes are an upper bound).
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
