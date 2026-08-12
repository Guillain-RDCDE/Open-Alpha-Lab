# Study 872 — Nominal-Price Illusion 🪙

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do low nominal-price names carry the lottery look & under-earn (Kumar; Birru-Wang)? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The money-illusion premium **fails to replicate — and inverts** — on 50 liquid US mega-caps. The specified long-cheap / short-dear spread is **+2.92 bps/day** (Newey-West *t* = **+3.01**): the low-priced names actually **out-earned** the expensive ones (2010–2026), with a *higher* Sharpe (**+1.15 vs +0.82**). It is significant but *opposite in sign* to the claim, holds in both eras (*t* = +2.11 / +2.16), sits ≈**2.4σ into the right tail** of a 1,000-permutation placebo, and a 20-seed synthetic control recovers a *planted* under-earn relation cleanly (fires on **0/20** nulls). The retail-lottery segment (penny / low-dollar names) is **absent** from a mega-cap survivor panel — *honest low power* — and where the illusion should bite, the cross-section instead rewards its lower-dollar value names. *Survivorship + adjusted-price proxy: current-membership mega-caps (rarely cheap), Close split-back-adjusted — named on the Signal axis.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Even the data-mined *sign-flip* book (long cheap / short dear) is a mirage: net **+0.79 bps/day** at 1 bp one-way but *insignificant* (*t* = +0.76), and **−7.21 bps/day** at 5 bps. No version survives the **2.14 bps/day** round-trip friction. |

> **In one sentence:** the celebrated nominal-price illusion — cheap-looking stocks should be
> over-priced lotteries that under-earn — **does not survive on liquid US mega-caps**; here the
> relation is significant but *reversed* (the low-priced names out-earned with a *higher* Sharpe,
> NW *t* = +3.01), and no version of the book survives costs, so the honest read is **claimed
> signal absent, paycheck a mirage**.

## What we tested

Kumar (2009), **"Who Gambles in the Stock Market?"** & Birru-Wang (2016), **"Nominal Price
Illusion"**: the raw **nominal share price** ($10 vs $500) is a pure money-illusion
characteristic — it carries no value information — yet retail lottery demand clusters in
low-priced names, which *should* make them over-priced lotteries (higher vol / right skew, lower
risk-adjusted returns). We take the cross-sectional version on a **liquid 50-name US
cross-section (yfinance daily OHLC, total-return, 2010-01-04 → 2026-06-30)**: sort on the
**nominal price level** (adjusted Close, a proxy), point-in-time (price known at the close of
`t−1`, one shift, zero look-ahead), long the cheapest 30% / short the priciest 30%, with a
Newey-West *t* on the daily `lo − hi` spread, each book's vol / skew / Sharpe, a 1,000-permutation
placebo, a two-era robustness cut, a costed timer, and a 20-seed synthetic positive control. The
universe is a **current-membership** survivor set (`quantlab.universe` opt-in guard) that is
*rarely cheap*, and the Close is *split-back-adjusted* — both named on the **Signal** axis
(honest low power). **Dedup:** [11-vanishing-penny](../11-vanishing-penny/) studies the literal
**penny-stock** tail, not a continuous price-level sort of liquid names;
[365-lottery-max-effect](../365-lottery-max-effect/) sorts on the realized **MAX** return, not
the price *level*; [250-reverse-split](../250-reverse-split/) is the **event** of a price-level
reset, not a standing cross-sectional sort; [93-round-numbers](../93-round-numbers/) is
**round-number** magnetism (a within-name barrier), not a cross-name cheap-vs-dear sort. As-of
**2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a cheap-looking share *should* be an over-priced lottery — and why on mega-caps the cheap names quietly out-earned instead |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the spread Newey-West *t*, the per-book vol/skew/Sharpe read, the 1,000-permutation placebo, the two-era cut, the cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`nominal_price/`](nominal_price/). Cross-section pulled through the
`quantlab.universe` survivorship guard (current membership, rarely cheap → magnitudes are an
upper bound; Close is split-back-adjusted, a nominal-price proxy). **Not investment advice** —
research & education. See [LICENSE](../../LICENSE).*
