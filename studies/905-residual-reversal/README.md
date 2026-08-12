# Study 905 — Residual Reversal ↩️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does factor-cleaning rescue weekly reversal on mega-caps? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The market-model **residual** reversal — long past-week residual losers, short winners, on the top-60% by dollar volume — earns **−0.38 bps/week** (Newey-West *t* = **−0.05**): a flat line. The **raw** foil is equally dead (−0.03 bps), dropping the screen only reaches +2.31 bps at *t* = +0.32, the 1,000-permutation placebo sits dead-centre (*p* = 0.51), and the two eras **flip sign** (*t* = +0.26 / −0.23). A 20-seed synthetic control recovers a *planted* residual reversal cleanly (*t* = +24.99, above the factor-muddied raw *t* = +13.23; fires on **0/20** nulls) — so the cleaner works and the null is real: short-term reversal is a small/illiquid-breadth effect, absent in 50 liquid mega-caps. *Survivorship: current-membership mega-caps — magnitudes are an upper bound, and still zero.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | A ~zero gross edge cannot survive the **2.96 bps/week** round-trip friction of a fully-turning weekly book: net **−3.35 bps/wk** at 1 bp one-way, **−11.35** at 5 bps. Even the un-screened +2.31 bps/week gross is eaten several times over. |

> **In one sentence:** the celebrated fix for reversal — reverse on the market-model
> **residual** to dodge bid-ask bounce and factor moves — cleans the signal beautifully in
> a planted synthetic world, but on 50 liquid US mega-caps there is **no weekly reversal
> to clean** (residual *and* raw sit at zero, placebo dead-centre, eras sign-flipping), and
> costs turn the flat gross deeply negative: **Signal None, paycheck a Mirage**.

## What we tested

Blitz, Huij, Lansdorp & Verbeek (2013), **"Short-Term Residual Reversal"**: the raw
one-week reversal (buy last week's losers, sell winners) mostly harvests **bid-ask bounce**
and **common-factor** moves, so it dies at the spread; regress each name's weekly return on
the market, keep the **residual**, and reverse on that instead, on a liquid subset. We run
the self-contained version on a **liquid 50-name US cross-section (yfinance daily OHLC +
Volume, total-return, 2010-01-04 → 2026-06-30 → 861 weekly returns)**: each name's
**weekly market-model residual** (trailing-52-week rolling OLS on the equal-weight market),
sorted point-in-time (signal known at the close of `w−1`, one shift, zero look-ahead) on
the **top 60% by dollar volume**, long bottom-30% / short top-30%, with the **raw** weekly
reversal beside it as the foil, a Newey-West *t*, a 1,000-permutation placebo, a two-era
cut, a costed timer, and a 20-seed synthetic positive control. The universe is a
**current-membership** survivor set (`quantlab.universe` opt-in guard) — named on the
**Signal** axis. **Dedup:** [329-one-month-reversal](../329-one-month-reversal/) is the
**monthly raw** reversal (no cleaning); [800-high-frequency-reversal](../800-high-frequency-reversal/)
is the **daily/intraday raw** version; [377-bid-ask-bounce](../377-bid-ask-bounce/) studies
the **bounce** itself (here the contaminant, not the signal); [237-residual-momentum](../237-residual-momentum/)
is residual **momentum** — continuation over a long window, the opposite sign at the
opposite horizon. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why cleaning the factor out of reversal *should* help — and why on mega-caps there was nothing to clean |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the residual-vs-raw race, the spread Newey-West *t*, the pooled Welch book test, the 1,000-permutation placebo, the two-era cut, the cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`resid_reversal/`](resid_reversal/). Cross-section pulled through the
`quantlab.universe` survivorship guard (current membership → magnitudes are an upper bound).
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
