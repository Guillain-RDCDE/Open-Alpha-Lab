# Study 884 — Convexity Barbell 🏋️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a duration-matched SHY+TLT barbell out-earn the IEF bullet on its extra convexity? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | It does **not**. The barbell (0.605·SHY + 0.395·TLT, duration-matched to IEF) earns **+2.15%/yr vs the bullet's +2.28%** at the same 6.4% vol and −23.5% drawdown; the daily spread is **−0.054 bps** (Newey-West *t* = **−0.27**), its bootstrap CI straddles zero **[−0.43, +0.34]**, the excess-vs-excess Sharpe advantage is **−0.02**, and it is flat in both eras. The convexity is genuinely *there* analytically but invisible in total return: the `f²` slope is **wrong-signed (−0.22)**, the convexity smile is absent, and in **2022's historic selloff** — the exact "big move" the claim needs — the barbell **under-performed** (−15.40% vs −15.16%). The market prices convexity into the wings' lower yield and charges butterfly (curve-reshaping) risk the bullet avoids. A 20-seed synthetic control recovers a *planted* under-priced-convexity edge cleanly (spread *t* = +4.3, fires on 0/20 nulls), so the machinery is sound — the net edge is simply absent. *Ladder selection named on the Signal axis.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | There is no gross edge to cost. The barbell turns over slowly (frictions are tiny), but the spread is ≈ 0 before costs and negative after (net ~**−0.14%/yr**). The "free convexity lunch" is fully offset by carry give-up and curve risk — a **Mirage**. |

> **In one sentence:** a barbell really is more convex than a duration-matched bullet, but
> on the Treasury tape that convexity is exactly paid for — a lower yield plus exposure to
> curve reshaping — so the duration-matched SHY+TLT barbell **earns slightly less** than the
> IEF bullet and even *lost* to it in 2022's record rate move: **claimed edge absent,
> paycheck a mirage.**

## What we tested

The textbook structure (Fabozzi; Ilmanen): a **duration-matched barbell** carries more
**convexity** than the **bullet**, so `+½·C·(Δy)²` should make it out-earn the bullet when
yields move a lot. We rebuild it from three liquid iShares Treasury ETFs + a cash leg —
**bullet = IEF (7-10y)**, **barbell = w·SHY (1-3y) + (1-w)·TLT (20y+)** with `w` set each
day so the barbell's empirical duration (its trailing-252d beta to a rates factor, known at
the close of `t−1`, one shift, zero look-ahead) equals IEF's — plus **BIL** cash for the
excess-vs-excess Sharpe race (yfinance daily total-return closes, 2010-01-04 → 2026-06-30).
We compare total return, Sharpe, drawdown, a convexity regression + smile, a calendar-year
table, a two-era cut, a bootstrap CI, a leg-permutation placebo, a costed rebalancing timer,
and a 20-seed synthetic positive control. The three-bond ladder is a **design selection** —
named on the **Signal** axis. **Dedup:** [59-downhill](../59-downhill/) and
[380-curve-roll-down](../380-curve-roll-down/) test the first-order **roll-down/carry** trade
(ageing down a sloped curve), not a second-order convexity compare;
[826-treasury-duration-bab](../826-treasury-duration-bab/) runs a **levered, beta-neutral**
long-short across maturity buckets, not a long-only duration-*matched* barbell-vs-bullet;
[581-term-premium](../581-term-premium/) is a **directional** when-to-own-duration timer,
whereas this study takes no duration view — barbell and bullet are duration-matched.
As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a barbell *should* be a free convexity lunch — and why on the Treasury tape it isn't (the barbell earns *less* than the bullet, and lost to it in 2022) |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the duration ladder & match, the spread's Newey-West *t* and bootstrap CI, the convexity regression + smile, the 2022 tell, the two-era cut, the leg-permutation placebo, the costed timer, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`barbell/`](barbell/). SHY/IEF/TLT + BIL total-return closes pulled via yfinance
into this study's own `_cache/`; the reproducible core runs offline.
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
