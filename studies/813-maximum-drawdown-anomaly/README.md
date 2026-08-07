# Study 813 — Maximum-Drawdown Anomaly 📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do deep-drawdown names under-earn (distress) or rebound? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | Sorting 50 liquid US mega-caps on their **trailing 12-month maximum drawdown**, the long-calm / short-distressed spread is **−4.35 bps/day** (Newey-West *t* = **−2.36**): the negative sign means the **distressed** names *out-earned* — a **drawdown reversal**, one of the two outcomes the claim entertained. A 1,000-permutation placebo confirms it isn't a lucky sort (≈**4.2σ** into the left tail). But it is **not robust across eras** — *t* = **−1.10** in 2010–2017 vs **−2.09** in 2018–2026 — so it clears the pooled \|t\| ≥ 2 bar only marginally and on one half of the sample. A 20-seed synthetic control recovers a *planted distress* relation cleanly (*t* = +9.03) and stays quiet on the null, so the sign is genuine. *Survivorship: current-membership mega-caps — the deepest drawdowns (permanent losers) are absent, so magnitudes are an upper bound.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The specified book loses money (**−6.48 bps/day** net at 1 bp one-way). Even the *profitable* rebound direction (long distressed) earns only **+2.21 bps/day** net at a fantasy 1 bp — *t* = +1.19, **not significant** — and turns negative (−5.79) by 5 bps. No paycheck either way. |

> **In one sentence:** on liquid US mega-caps the deepest-drawdown names don't keep sinking —
> they **rebound** (long-calm / short-distressed spread −4.35 bps/day, NW *t* = −2.36), but the
> effect lives in a single era and no version survives costs, so the honest read is **a fragile
> reversal, not a distress premium, and a mirage to trade**.

## What we tested

The **maximum-drawdown anomaly**: sort a cross-section on each name's **trailing 12-month
maximum drawdown** — the largest peak-to-trough decline of its cumulative total return — and ask
whether the recently distressed (deep-drawdown) names subsequently **under-earn** (a distress
premium) or **rebound** (reversal). We take **no prior on the sign**. We use a **liquid 50-name
US cross-section (yfinance daily OHLC, total-return, 2010-01-04 → 2026-06-30)**: each name's
trailing-252-day maximum drawdown (vectorised sliding-window running peak), sorted point-in-time
(signal known at the close of `t−1`, one shift, zero look-ahead), long the calm bottom 30% /
short the distressed top 30%, with a Newey-West *t* on the daily spread, a 1,000-permutation
placebo, a two-era robustness cut, a costed long-short timer in both directions, and a 20-seed
synthetic positive control. The universe is a **current-membership** survivor set
(`quantlab.universe` opt-in guard) — named on the **Signal** axis, and it bites hard here since
the deepest drawdowns of all (permanent losers) are exactly what a survivor panel deletes.
**Dedup:** [333-recovery-speed](../333-recovery-speed/) measures how *fast* a name recovers, not
drawdown **depth**; [816-drawdown-duration](../816-drawdown-duration/) measures how *long* a name
is underwater (the horizontal axis), not the **vertical** depth here;
[540-distress-risk](../540-distress-risk/) uses a **fundamental** default score, not a price-only
drawdown; [332-downside-beta](../332-downside-beta/) uses a name's **beta in down markets**, not
its own peak-to-trough decline. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | distress vs reversal in one picture — and why on mega-caps the wounded *bounced* |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the spread Newey-West *t*, the pooled Welch book test, the 1,000-permutation placebo, the two-era cut, the cost math in both directions, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`max_drawdown/`](max_drawdown/). Cross-section pulled through the
`quantlab.universe` survivorship guard (current membership → magnitudes are an upper bound).
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
