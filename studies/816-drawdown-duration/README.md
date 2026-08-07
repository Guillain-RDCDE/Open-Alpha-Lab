# Study 816 — Drawdown Duration ⏱️📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the *fraction of the year a name spent underwater* predict its forward return? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | On 50 liquid US mega-caps, sorting on **trailing-252-day time-underwater** (fraction of the year the cumulative return sat below its running high-water mark) predicts **nothing**. The long-high-underwater / short-low-underwater spread is **−1.12 bps/day** (Newey-West *t* = **−0.80**) — statistically zero, ≈**1.12σ** from a 1,000-permutation null (two-sided p = **0.26**), with a **sign that flips between eras** (+0.28 / −1.17) and a clean 20-seed synthetic control. The market neither **pays** a persistent-drawdown premium nor reliably **keeps sinking** the underwater names. *Survivorship: current-membership mega-caps — the names that stayed underwater and died are absent, so any "losers keep sinking" tilt is understated.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The specified book loses money gross (−1.12 bps/day) and net (**−3.26 bps/day** at 1 bp one-way, −11.26 at 5 bps). Even the sign-flip (buy the fresh-high names) earns only +1.12 bps/day gross — less than the **2.14 bps/day** round-trip friction at a mere 1 bp. A Mirage in either direction. |

> **In one sentence:** how *long* a mega-cap spent underwater over the past year — its
> persistent-drawdown risk — **does not price** in the cross-section here; the long-short spread
> is a statistical zero (NW *t* = −0.80) whose sign even flips between eras, and no version of the
> book survives costs, so the honest read is **no signal, no paycheck**.

## What we tested

The drawdown curve has two moments: **depth** (how far a name fell) and **duration** (how long it
stayed down). This study takes the *duration* side as **time-underwater** — the fraction of the
trailing year each name's cumulative total return sat **below its running high-water mark** — and
asks whether the market **pays** for bearing persistent-drawdown names (a positive long-high /
short-low spread) or whether they simply **keep sinking** (a negative spread). We run it on a
**liquid 50-name US cross-section (yfinance daily OHLC, total-return, 2010-01-04 → 2026-06-30)**:
each name's **trailing-252-day time-underwater** (vectorised `cumprod` → `cummax` → underwater
indicator → rolling mean), sorted point-in-time (signal known at the close of `t−1`, one shift,
zero look-ahead), with a Newey-West *t* on the daily spread, a 1,000-permutation two-sided
placebo, a two-era robustness cut, a costed long-short timer, and a 20-seed synthetic positive
control. The universe is a **current-membership** survivor set (`quantlab.universe` opt-in guard)
— named on the **Signal** axis. **Dedup:** [813-max-drawdown](../813-max-drawdown/) tests the
drawdown **depth**, not its **duration**; [333-recovery-speed](../333-recovery-speed/) tests how
**fast** a name recovers *after* a drawdown, not the *share of time* underwater;
[330-low-volatility](../330-low-volatility/) tests return **volatility**, a moment of the return
distribution, not a path-dependent drawdown statistic. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "time underwater" means, and why persistent-drawdown mega-caps neither paid a premium nor kept sinking |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the spread Newey-West *t*, the pooled Welch book test, the 1,000-permutation two-sided placebo, the two-era cut, the cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`drawdown_duration/`](drawdown_duration/). Cross-section pulled through the
`quantlab.universe` survivorship guard (current membership → magnitudes are an upper bound).
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
