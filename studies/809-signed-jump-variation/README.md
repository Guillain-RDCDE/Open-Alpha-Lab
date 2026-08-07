# Study 809 — Signed Jump Variation ⚡📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do downside-dominated names (negative signed jump) carry a premium (Bollerslev-Li-Zhao)? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The good-vol/bad-vol signed-jump premium **fails to replicate** on 50 liquid US mega-caps. The specified long-low-SJ / short-high-SJ spread is **−1.71 bps/day** (Newey-West *t* = **−1.36**, \|*t*\| < 2) — **insignificant** *and* **wrong-signed**: the upside-dominated ("good" vol, high-SJ) names, if anything, *out-earned* the downside names (2010–2026). It is flat in both eras (*t* = −0.26 / −1.44), sits only ≈**1.9σ into the left tail** of a 1,000-permutation placebo, and a 20-seed synthetic control recovers a *planted* Bollerslev-Li-Zhao relation cleanly (fires on **0/20** nulls) — so the null result is a genuine *absence*, not a broken engine. The signed-jump premium is a **smaller-cap** phenomenon; mega-caps are exactly where it should not appear. *Survivorship: current-membership mega-caps — magnitudes are an upper bound.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The specified book loses money net at any cost (**−3.85 bps/day** at 1 bp one-way, −11.85 at 5 bps), and the gross edge is not even reliably present. Even the data-mined *sign-flip* (long high-SJ) earns only +1.71 bps/day gross, which the **2.14 bps/day** round-trip friction at a mere 1 bp already eats — a Mirage in either direction. |

> **In one sentence:** the celebrated good-volatility/bad-volatility signed-jump premium —
> downside-dominated names should out-earn — **does not survive on liquid US mega-caps**; here
> the spread is insignificant and mildly *reversed* (NW *t* = −1.36), and no version of the book
> survives costs, so the honest read is **claimed signal absent, paycheck a mirage**.

## What we tested

Barndorff-Nielsen, Kinnebrock & Shephard (2010) realized **semivariance** and Bollerslev, Li &
Zhao (2020), **"Good Volatility, Bad Volatility, and the Cross Section of Stock Returns"**: split a
name's recent realized variance into upside `RS+ = Σ r²·1(r>0)` and downside `RS- = Σ r²·1(r<0)`;
the **signed jump variation** `SJ = (RS+ − RS-)/RV` is priced *negatively* (downside-dominated
names carry a premium), so a long low-SJ / short high-SJ book should earn a *positive* spread. We
take the self-contained daily version on a **liquid 50-name US cross-section (yfinance daily OHLC,
total-return, 2010-01-04 → 2026-06-30)**: each name's **trailing-21-day signed jump variation** of
daily returns (vectorised sign-split of `r²`, scaled to `[-1,+1]`), sorted point-in-time (signal
known at the close of `t−1`, one shift, zero look-ahead), with a Newey-West *t* on the daily spread,
a 1,000-permutation placebo, a two-era robustness cut, a costed long-short timer, and a 20-seed
synthetic positive control. The universe is a **current-membership** survivor set
(`quantlab.universe` opt-in guard) — named on the **Signal** axis. **Dedup:**
[803-realized-skewness-reversal](../803-realized-skewness-reversal/) sorts on the standardised
**third moment** (`m3/m2^1.5`), not the **sign-split of variance**;
[505-left-tail-momentum](../505-left-tail-momentum/) carries a single **worst-return** order
statistic forward, not the up-vs-down **variance ratio**; [130-variance-risk-premium](../130-variance-risk-premium/)
is the market-level **implied-minus-realized** variance gap, not a cross-sectional realized split.
As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "bad" (downside) volatility *should* be paid a premium — and why on mega-caps the good-vol names held up instead |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the spread Newey-West *t*, the pooled Welch book test, the 1,000-permutation placebo, the two-era cut, the cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`signed_jump/`](signed_jump/). Cross-section pulled through the `quantlab.universe`
survivorship guard (current membership → magnitudes are an upper bound). **Not investment advice**
— research & education. See [LICENSE](../../LICENSE).*
