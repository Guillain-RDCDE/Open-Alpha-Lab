# Study 870 — Industry-Leader Lead-Lag 👑

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the biggest name in a sector lead the rest (Hou 2007)? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The industry-leader lead-lag **fails to replicate** on 50 liquid US mega-caps. The specified long-up-leader / short-down-leader followers spread is **−3.64 bps/week** (Newey-West *t* = **−0.77**) — statistically indistinguishable from zero, and on the *wrong* side if anything (followers earned slightly *more* after their leader **fell**, Welch *t* = −1.99). It is unremarkable against a 1,000-permutation placebo (right-tail *p* = **0.93**), **sign-flips across eras** (−13.96 bps early / +5.96 late), and is unchanged when leaders are re-designated by dollar volume (−3.87 bps). A 20-seed synthetic control recovers a *planted* diffusion emphatically (*t* = +20.79, fires on **0/20** nulls) — so the sort works; there is simply no lag to harvest. Slow within-industry diffusion is a **small-and-illiquid-firm** effect; mega-caps price sector news near-simultaneously. *Survivorship + static largest-cap leaders — magnitudes are an upper bound.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The specified book loses money gross and net (**−6.60 bps/week** at 1 bp one-way, −14.60 at 5 bps). Even the data-mined *sign-flip* earns only +3.64 bps/week gross, which the **2.96 bps/week** weekly round-trip friction at a mere 1 bp already eats — a Mirage in either direction. |

> **In one sentence:** the celebrated industry-leader lead-lag — the biggest firm's move
> should foreshadow its smaller peers' — **does not survive on liquid US mega-caps**; the
> spread is a flat −3.64 bps/week (NW *t* = −0.77), placebo-unremarkable, era-unstable, and
> no version of the book survives costs, so the honest read is **claimed signal absent,
> paycheck a mirage**.

## What we tested

Kewei Hou (2007), **"Industry Information Diffusion and the Lead-Lag Effect in Stock
Returns"**: information diffuses *within an industry* from the biggest name outward, so the
**largest-cap** firm's return this week should predict its smaller **followers'** return
next week. We take the self-contained weekly version on a **liquid 50-name US cross-section
(yfinance daily OHLCV, total-return, 2010-01-04 → 2026-06-30)** split into 8 GICS-style
sectors: designate the largest-cap of each as the **leader** (AAPL, GOOGL, AMZN, WMT, JPM,
JNJ, XOM, GE), read the leader's week-`w` return (point-in-time, one week of execution lag,
zero look-ahead), long the followers whose leader rose and short those whose leader fell,
with a Newey-West *t* on the weekly spread, a 1,000-permutation placebo, a two-era cut, a
dollar-volume leader re-designation, a costed long-short timer, and a 20-seed synthetic
positive control. The universe is a **current-membership** survivor set with **static
largest-cap leaders** (`quantlab.universe` opt-in guard) — named on the **Signal** axis.
**Dedup:** [379-etf-lead-lag](../379-etf-lead-lag/) tests a **basket (ETF) → member** lag, not
a **firm → firm within-industry** one; [506-industry-momentum](../506-industry-momentum/) sorts
**industries against each other**, not names *within* one; [538-industry-relative-reversal](../538-industry-relative-reversal/)
is a name's **own** deviation *reversing*, not a leader predicting a *different* name; and
[810-price-delay](../810-price-delay/) measures a name's delay in loading the **market** factor,
not one industry name leading another. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the biggest firm *should* lead its industry — and why on mega-caps there is no lag left to harvest |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the weekly spread Newey-West *t*, the pooled Welch leg test, the 1,000-permutation placebo, the two-era + dollar-volume robustness cuts, the cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`leader_lag/`](leader_lag/). Cross-section pulled through the `quantlab.universe`
survivorship guard (current membership + static largest-cap leaders → magnitudes are an upper
bound). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
