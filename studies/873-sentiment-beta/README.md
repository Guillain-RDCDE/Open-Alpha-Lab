# Study 873 — Sentiment Beta 🎭

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do high-sentiment-beta names go on to earn *less* (Baker-Wurgler)? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The famous sentiment-beta premium **fails to replicate — and inverts** — on 50 liquid US mega-caps. The specified long-low-beta / short-high-beta spread is **−6.20 bps/day** (Newey-West *t* = **−2.86**): the high-sentiment-beta names (the momentum tech mega-caps that sync with the speculative leg) actually *out-earned* the low-beta ones (2010–2026), and *more so* after sentiment peaked (−9.21 bps vs −4.91). It is significant but *opposite in sign* to the claim, sits ≈**4.71σ into the left tail** of a 1,000-permutation placebo, and a 20-seed synthetic control recovers a *planted* Baker-Wurgler relation cleanly (fires on **0/20** nulls) — so the sign-reversal is real, not machinery. The sentiment-beta effect is a **hard-to-value / speculative-small-stock** phenomenon; mega-caps are exactly where it should not appear. *Survivorship: current-membership mega-caps — magnitudes are an upper bound.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The specified book loses money gross and net (**−8.34 bps/day** at 1 bp one-way, −16.34 at 5 bps). Even the data-mined *sign-flip* (long high-beta) is a heavily factor-exposed ≈19%-vol long-short earning only +6.20 bps/day gross, which the **2.14 bps/day** round-trip friction at a mere 1 bp already eats a third of — a Mirage in either direction. |

> **In one sentence:** the celebrated Baker-Wurgler sentiment-beta premium — high-sentiment-beta
> names should under-earn — **does not survive on liquid US mega-caps**; here the relation is
> significant but *reversed* (the speculative-co-moving names out-earned, NW *t* = −2.86), and no
> version of the book survives costs, so the honest read is **claimed signal absent, paycheck a mirage**.

## What we tested

Baker & Wurgler (2006, 2007), **"Investor Sentiment and the Cross-Section of Stock Returns"**: the
stocks whose returns **co-move most with market sentiment** (high *sentiment beta*) are the
speculative, hard-to-value names that get over-priced in euphoria and **under-perform afterwards**,
so a long-low-beta / short-high-beta book should earn a *positive* spread. We take the self-contained
daily version on a **liquid 50-name US cross-section (yfinance daily OHLCV, total-return,
2010-01-04 → 2026-06-30)**: we proxy sentiment with a **tradable high-minus-low-volatility spread**
built from the panel itself (speculative tercile minus safe tercile — real data), estimate each
name's **252-day sentiment beta** to it (rolling `cov/var`), sort point-in-time (beta known at the
close of `t−1`, one shift, zero look-ahead), and measure the forward return of the long-low-beta /
short-high-beta book — with a Newey-West *t*, a **post-peak conditional** cut, a 1,000-permutation
placebo, a two-era robustness cut, a costed long-short timer, and a 20-seed synthetic positive
control. The universe is a **current-membership** survivor set (`quantlab.universe` opt-in guard) —
named on the **Signal** axis. **Dedup:** [258-baker-wurgler](../258-baker-wurgler/) tests the
**time-series / aggregate** sentiment-level → market-return claim, not the cross-sectional beta sort;
[255-fear-greed-index](../255-fear-greed-index/) is a market-timing gauge, not a beta sort;
[501-idiosyncratic-volatility](../501-idiosyncratic-volatility/) and
[330-low-volatility-anomaly](../330-low-volatility-anomaly/) sort on a name's **own volatility
level**, whereas sentiment beta is the **co-movement** with a sentiment *time series* (a high-vol name
uncorrelated with the speculative leg has a *low* sentiment beta). As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why co-moving with euphoria *should* signal lower future returns — and why on mega-caps the opposite happened |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the spread Newey-West *t*, the pooled Welch book test, the post-peak conditional, the 1,000-permutation placebo, the two-era cut, the cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`sentiment_beta/`](sentiment_beta/). Cross-section pulled through the
`quantlab.universe` survivorship guard (current membership → magnitudes are an upper bound).
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
