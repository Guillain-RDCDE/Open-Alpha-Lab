# Study 806 — Prospect-Theory Value 🧠🎰

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a high prospect-theory (TK) value predict *lower* future returns (Barberis-Mukherjee-Wang)? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | The famous prospect-theory-value premium **replicates with the predicted sign** on 50 liquid US mega-caps. The long-low-TK / short-high-TK spread is **+138.70 bps/month** (Newey-West *t* = **+3.15**): the boring **low-TK** names out-earned the lottery-like **high**-TK names (2015–2026), exactly as prospect theory predicts. It holds in both eras (*t* = +2.00 / +2.48), sits ≈**4.09σ into the right tail** of a 1,000-permutation placebo, and a 20-seed synthetic control recovers a *planted* relation cleanly (fires on **0/20** nulls). *Survivorship: current-membership mega-caps — magnitudes are an upper bound.* |
| **Tradability** — can you get paid for it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | The net edge **survives** conservative costs (**+124.53 bps/month** at 5 bps one-way, *t* = +2.57, ≈+14.9%/yr) — so not a Mirage — but the magnitude is a **survivorship upper bound** (the short leg's blown-up lottery names are absent) and the 50-name universe concentrates the short into a few hard-to-borrow lottery mega-caps whose realistic borrow/squeeze exceeds the 50 bps/yr charged. Real signal, **fragile** paycheck. |

> **In one sentence:** the celebrated prospect-theory-value effect — stocks that look like a
> good gamble under Tversky-Kahneman are over-priced and under-earn — **does replicate on
> liquid US mega-caps** (long-low-TK / short-high-TK = +138.70 bps/month, NW *t* = +3.15), and
> the spread even clears conservative costs, but it leans on a survivorship-inflated short leg,
> so the honest read is **claimed signal Real, paycheck Fragile**.

## What we tested

Barberis, Mukherjee & Wang (2016), **"Prospect Theory and Stock Returns: An Empirical Test"**:
for each stock compute the **cumulative-prospect-theory (TK) value** a Tversky-Kahneman investor
would place on its recent return distribution; the theory says high-TK (lottery-like) names are
over-priced and under-earn, so a long low-TK / short high-TK book should earn a *positive*
spread. We take the self-contained daily version on a **liquid 50-name US cross-section (yfinance
daily OHLC, total-return, 2010-01-04 → 2026-06-30)**: each name's **trailing-1,260-day (≈5y) TK
value** (exact TK value function + inverse-S probability weighting, `γ=0.61`/`δ=0.69`), sorted
**monthly, point-in-time** (signal known at the close of month `t`, hold month `t+1`, full window
required, zero look-ahead), with a Newey-West *t* on the monthly spread, a 1,000-permutation
placebo, a two-era robustness cut, a costed long-short timer, and a 20-seed synthetic positive
control. The universe is a **current-membership** survivor set (`quantlab.universe` opt-in guard)
— named on the **Signal** axis. **Dedup:** [365-lottery-max-effect](../365-lottery-max-effect/)
uses the single **MAX** daily return, not the full probability-weighted TK value;
[327-disposition-overhang](../327-disposition-overhang/) tests the **capital-gains overhang**
(a reference-point holding story), not the past return *distribution* as a gamble;
[503-expected-idiosyncratic-skewness](../503-expected-idiosyncratic-skewness/) tests **modelled
ex-ante** skewness, not the full Tversky-Kahneman functional. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a lottery-like tape *should* signal lower future returns — and why here it did |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the TK value math, the spread Newey-West *t*, the pooled Welch book test, the 1,000-permutation placebo, the two-era cut, the cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`prospect_theory/`](prospect_theory/). Cross-section pulled through the
`quantlab.universe` survivorship guard (current membership → magnitudes are an upper bound).
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
