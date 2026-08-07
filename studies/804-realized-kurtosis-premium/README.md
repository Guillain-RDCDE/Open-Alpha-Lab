# Study 804 — Realized-Kurtosis Premium 🎲📊

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do fat-tailed (high-realized-kurtosis) names earn a premium (Amaya et al)? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The source paper itself calls realized **kurtosis** a **weak / ambiguous** predictor, mostly subsumed by skewness and volatility — and that is exactly what we find. On 50 liquid US mega-caps the specified long-high-kurt / short-low-kurt spread is **+1.78 bps/day** — the *right sign*, but Newey-West *t* = **+1.79** (below the \|t\| ≥ 2 bar), the pooled book Welch *t* is a limp **+0.69**, and the effect is a flat **zero** in 2010–2017 (*t* = +0.11), surfacing only marginally in 2018–2026 (*t* = +2.12). A 1,000-permutation placebo puts the spread ~**+2.2σ** into the right tail (p = 0.016), but that test ignores the serial correlation the NW *t* accounts for — the honest robust read fails to clear 2. The 20-seed synthetic control is clean (fires **0/20** on the null, recovers a planted relation at *t* = +12.81). *Survivorship: current-membership mega-caps — magnitudes are an upper bound.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The +1.78 bps/day gross edge is already **smaller than the 2.14 bps/day round-trip friction** at a mere 1 bp one-way — net **−0.35 bps/day** — and collapses to **−8.35 bps/day** at 5 bps. No version of the book survives costs. |

> **In one sentence:** the realized-kurtosis premium is the *weak* sibling of realized skewness
> in the very same paper, and on liquid US mega-caps it stays weak — a right-signed but
> sub-threshold **+1.78 bps/day** (NW *t* = +1.79) that lives entirely in the recent era and is
> eaten by costs before it can be traded.

## What we tested

Amaya, Christoffersen, Jacobs & Vasquez (2015), **"Does Realized Skewness Predict the
Cross-Section of Equity Returns?"** — the same paper whose headline is realized *skewness* also
tests realized **kurtosis** (a name's recent fat-tailedness, the fourth standardised moment) and
reports it as a **weak / ambiguous** predictor. We take the self-contained daily version on a
**liquid 50-name US cross-section (yfinance daily OHLC, total-return, 2010-01-04 → 2026-06-30)**:
each name's **trailing-21-day realized kurtosis** of daily returns (population `m4 / m2**2`,
vectorised via rolling raw moments — not `rolling.apply`), sorted point-in-time (signal known at
the close of `t−1`, one shift, zero look-ahead), **long the top 30% (high kurt) / short the
bottom 30% (low kurt)**, with a Newey-West *t* on the daily spread, a 1,000-permutation placebo,
a two-era robustness cut, a costed long-short timer, and a 20-seed synthetic positive control.
The universe is a **current-membership** survivor set (`quantlab.universe` opt-in guard) — named
on the **Signal** axis. **Dedup:** [803-realized-skewness-reversal](../803-realized-skewness-reversal/)
tests the **third** moment (asymmetry), the paper's *strong* sibling — kurtosis is the *fourth*
(symmetric fat tails); [501-idiosyncratic-volatility](../501-idiosyncratic-volatility/) tests the
**second** moment (dispersion), whereas kurtosis is scale-free; [365-lottery-max-effect](../365-lottery-max-effect/)
uses the single **MAX** daily return, not the full fourth moment. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "fat tails" mean for a stock, why a kurtosis premium *might* exist — and why here it barely shows up |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the spread Newey-West *t*, the pooled Welch book test, the 1,000-permutation placebo, the two-era cut, the cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`realized_kurtosis/`](realized_kurtosis/). Cross-section pulled through the
`quantlab.universe` survivorship guard (current membership → magnitudes are an upper bound).
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
