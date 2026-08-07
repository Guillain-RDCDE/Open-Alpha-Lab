# Study 811 — Zero-Return Illiquidity 🕳️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do names that print *exactly-zero* returns earn an illiquidity premium (Lesmond-Ogden-Trzcinka)? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The premium is **absent** on 50 liquid US mega-caps. The specified long-high-zero / short-low-zero spread is **−1.37 bps/day** (Newey-West *t* = **−1.29**, |t| < 2) — insignificant and faintly *wrong*-signed, unstable across eras (*t* = −1.69 / −0.41), sitting only ≈1.35σ inside a 1,000-permutation null. The reason is baked in: the proxy is **near-degenerate** here — the *median* mega-cap prints a zero return on **0.00%** of trailing-year days (max 2.38%), because these names never sit still. A 20-seed synthetic control recovers a *planted* premium cleanly (*t* = +11.23, fires on **0/20** nulls), so the flat real result is a true absence, not a broken sort. *Survivorship: current-membership mega-caps — magnitudes are an upper bound.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The book loses money gross (−1.37 bps/day) and worse net (**−3.51 bps/day** at 1 bp, −11.51 at 5 bps). And the charged cost is optimistic: the long leg is *by construction* the least-liquid names, exactly where a 1 bp one-way is a fantasy floor. A Mirage in both directions. |

> **In one sentence:** the Lesmond-Ogden-Trzcinka zero-return illiquidity proxy — often-zero
> names should earn a premium — **has almost no signal to give on liquid mega-caps** (half of
> them never print a zero-return day), the spread is an insignificant −1.37 bps/day, and no
> version of the book survives costs, so the honest read is **signal None, paycheck a mirage**.

## What we tested

Lesmond, Ogden & Trzcinka (1999), **"A New Estimate of Transaction Costs"**: when the
round-trip cost of trading exceeds the day's information the price does not move, so the
**frequency of exactly-zero daily returns** proxies a name's transaction cost / illiquidity;
illiquid names should earn a premium (Amihud & Mendelson 1986), so a long-high-zero /
short-low-zero book should earn a *positive* spread. We take the price-only reduced form on a
**liquid 50-name US cross-section (yfinance daily OHLC, total-return, 2010-01-04 →
2026-06-30)**: each name's **trailing-252-day proportion of exactly-zero (`|r|<1e-8`) daily
returns**, sorted point-in-time (signal known at the close of `t−1`, one shift, zero
look-ahead), long the top 30% (illiquid) / short the bottom 30% (liquid), with a Newey-West
*t* on the daily spread, a 1,000-permutation placebo, a two-era robustness cut, a costed
long-short timer, and a 20-seed synthetic positive control. The universe is a
**current-membership** survivor set (`quantlab.universe` opt-in guard) — named on the
**Signal** axis. **Dedup:** [140-amihud-illiquidity](../140-amihud-illiquidity/) uses the
**volume-scaled** Amihud ILLIQ (|return| per dollar of volume), not the price-only zero
count; [141-turnover](../141-turnover/) sorts on **turnover** (a volume/activity measure),
not a transaction-cost proxy; [812-corwin-schultz](../812-corwin-schultz/) backs a bid-ask
spread out of the intraday **high-low range**, not the close-to-close zero frequency. As-of
**2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a stock that sits still is an illiquidity meter — and why mega-caps almost never sit still |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the signal degeneracy, the spread Newey-West *t*, the pooled Welch book test, the 1,000-permutation placebo, the two-era cut, the cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`zero_return/`](zero_return/). Cross-section pulled through the `quantlab.universe`
survivorship guard (current membership → magnitudes are an upper bound).
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
