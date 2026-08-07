# Study 820 — Expected-Shortfall Premium 🎯📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do fat-left-tail (high-ES) names go on to earn *more* (priced tail risk)? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The claimed downside tail-risk premium **replicates in sign** on 50 liquid US mega-caps — the long-high-ES / short-low-ES spread is **+5.41 bps/day** (Newey-West *t* = **+2.80**), ≈**4.8σ** into a 1,000-permutation placebo, synthetic control clean — but it is **not robust**: era-dependent (*t* = +1.67 pre-2018 vs +2.26 post, significant in only one half), a sub-threshold pooled book test (Welch *t* = +1.84), and — since Expected Shortfall is ~collinear with volatility on a **survivor** universe — most honestly read as the surviving high-vol tech mega-caps winning the 2018–2026 melt-up (the *inverse* of the low-vol anomaly), not a clean priced tail premium. *Survivorship: current-membership mega-caps — the bias flatters this sort, so magnitudes are an upper bound.* |
| **Tradability** — can you get paid for it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | At an optimistic **1 bp** one-way the book nets **+3.27 bps/day** (~+8%/yr, Sharpe 0.42) — but net *t* = +1.64 (< 2), no robust edge; at a realistic **5 bps** the friction swamps it and the book is **−4.73 bps/day** (*t* = −2.37). The gross spread is real but too thin to survive costs. |

> **In one sentence:** the downside Expected-Shortfall premium **does replicate in sign** on
> liquid US mega-caps (long high-ES / short low-ES, +5.41 bps/day, NW *t* = +2.80), but it is an
> era-dependent, survivor-flattered *high-volatility* sort dressed as a tail premium, and it is
> **too thin to trade** once realistic friction is charged.

## What we tested

A priced downside **tail-risk premium**: sort stocks on their recent **historical Expected
Shortfall** (CVaR) at 5% — the mean of the worst 5% of daily returns — and, if the fat left tail
is genuine crash exposure, the high-ES names should be compensated with **higher** future
returns. We take the self-contained daily version on a **liquid 50-name US cross-section
(yfinance daily OHLC, total-return, 2010-01-04 → 2026-06-30)**: each name's **trailing-252-day
Expected Shortfall** (non-parametric, the mean of its worst `ceil(0.05·252)=13` daily returns),
sorted point-in-time (signal known at the close of `t−1`, one shift, zero look-ahead), long the
top 30% (high ES) / short the bottom 30% (low ES), with a Newey-West *t* on the daily spread, a
1,000-permutation placebo, a two-era robustness cut, a costed long-short timer, and a 20-seed
synthetic positive control. The universe is a **current-membership** survivor set
(`quantlab.universe` opt-in guard) — named on the **Signal** axis, and here the bias *flatters*
the sort. **Dedup:** [332-downside-beta](../332-downside-beta/) is the **systematic** beta on
down markets (a factor covariance), not a name's **own** tail depth;
[505-left-tail-momentum](../505-left-tail-momentum/) is the **continuation** of extreme negative
returns (tail dynamics), not the ES *level*; [501-idiosyncratic-volatility](../501-idiosyncratic-volatility/)
is the **whole two-sided dispersion**, whereas ES is one-sided (the worst days only). As-of
**2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a fat left tail *should* be paid — and why on survivor mega-caps this is really the high-vol names winning |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the spread Newey-West *t*, the pooled Welch book test, the 1,000-permutation placebo, the two-era cut, the cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`expected_shortfall/`](expected_shortfall/). Cross-section pulled through the
`quantlab.universe` survivorship guard (current membership → magnitudes are an upper bound).
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
