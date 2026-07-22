# Study 788 — Overnight / Intraday Tug of War 🌙🔀☀️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do past-overnight winners keep winning at night and reverse by day? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | Sorting a 50-name US cross-section on its trailing overnight return, the high-minus-low spread earns **+3.71 bps/day overnight** (persistence, Newey-West *t* = **+4.62**) and **−2.39 bps/day intraday** (reversal, NW *t* = **−2.36**) — the two legs pulling opposite ways, exactly Lou-Polk-Skouras. A 1,000-permutation placebo puts the overnight leg ≈6σ out (**p = 0.00000**); it holds in both sub-periods (overnight *t* = +4.32 then +3.11); a 20-seed synthetic control never fires on the null. Decisively, the legs roughly **cancel close-to-close** (+1.32 bps, *t* = +1.06) — a within-day redistribution, not a net premium. *Survivorship: current-membership mega-caps — magnitudes are an upper bound.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Capturing the overnight leg means entering the 2×-NAV book at the close and unwinding at the open — four spread-crossings a day. The gross edge is **+3.71 bps/day**, but at a mere **1 bp** one-way cost the round-trip friction (4.14 bps/day) **already exceeds it** (net **−0.43 bps/day**, *t* = −0.51); at 5 bps it is **−41%/yr**. A real signal you cannot get paid for. |

> **In one sentence:** stocks with high past overnight returns really do keep earning
> overnight (NW *t* = +4.62) while handing it back intraday (NW *t* = −2.36) — a genuine
> cross-sectional tug of war — but the two legs nearly cancel close-to-close and the
> overnight leg is far too small to survive the four-crossings-a-day cost of harvesting
> it, so the honest read is **real signal, mirage paycheck**.

## What we tested

The Lou, Polk & Skouras (2019) **"tug of war"**: sort stocks on their trailing
**overnight** (prev-close → open) return; the theory says the high-overnight names keep
earning overnight (persistence) while **reversing** intraday (open → close), and the
low-overnight names do the opposite — so a long-short spread should show a *positive
overnight leg* and a *negative intraday leg* that fight each other. We take it literally
on a **liquid 50-name US cross-section (yfinance daily OHLC, total-return, 2010-01-04 →
2026-06-30)**, decomposed night/day with the exact identity (`quantlab.decompose`), sorted
point-in-time (signal known at the close of `t−1`, one shift, zero look-ahead), with a
Newey-West *t* on each leg-spread, a 1,000-permutation placebo, a two-era robustness cut,
a costed overnight-capture timer (one-way × NAV per leg, shorts pay borrow), and a 20-seed
synthetic positive control. The universe is a **current-membership** survivor set
(`quantlab.universe` opt-in guard) — named on the **Signal** axis, so the magnitudes are
an upper bound. **Dedup:** [01-overnight-anomaly](../01-overnight-anomaly/) tests the
**aggregate / index-level** night-vs-day split, not the cross-sectional sort;
[640-gold-overnight](../640-gold-overnight/) runs the same decomposition on a **single
asset** (gold); [116-power-hour](../116-power-hour/) is an **intraday-clock** effect (the
last trading hour), not the overnight-vs-intraday cross-section. None run the
overnight-sorted persistence-vs-reversal tug — this study does. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "past-overnight winners keep winning at night but give it back by day" is a real, measurable pattern — and why it nets to almost nothing close-to-close and pays even less after costs |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the per-leg Newey-West splits, the pooled Welch book test, the 1,000-permutation placebo, the two-era robustness cut, the four-crossings-a-day cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`overnight_intraday_tug/`](overnight_intraday_tug/). Cross-section pulled
through the `quantlab.universe` survivorship guard (current membership → magnitudes are an
upper bound). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
