# Study 643 — Payrolls-Day-Effect 📰📆

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does SPY move systematically on NFP release mornings? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | Release-day return **+12.43 bps** vs +3.37 bps other days — Welch *t* = **+1.31**, Newey-West *t* = **+1.31**, two-sided random-calendar placebo **p = 0.079** over 20,000 draws. None clear the desk's **t ≥ 2** bar. Hit rate 58.4% (Wilson [53.2%, 63.4%]) barely clears SPY's own 54.1% baseline up-day rate. A nominal pre-release drift ([−3..−1] cumulative *t* = +2.07) is named as one uncorrected hit among seven offsets tested — a hint, not a certified finding. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The naive "own SPY only on NFP day" timer nets **+2.43 bps/event at 5 bps** (~+0.29%/yr from 12 trades) and turns **net negative at 10 bps** (−7.57 bps, ~−0.91%/yr); worst single release day **−6.0%**. No residual survives costs, and the point estimate was already uncertified. |
| **"Louder, not directional?"** | ![Confirmed](https://img.shields.io/badge/Louder%2C_not_directional%3F-Confirmed-8b949e?style=flat-square) | Realized SPY high-low range clears the bar decisively (**1.474%** vs 1.342%, Welch *t* = **+2.50**) — NFP mornings are genuinely, mechanically noisier. That loudness never converts into a certified directional edge (Signal *t* = 1.31). |

> **In one sentence:** on 353 actual NFP release mornings since 1997, SPY does move more (a
> real, mechanical +2.50-*t* realized-range bump — same resolution-of-scheduled-uncertainty
> signature as the desk's FOMC vol crush) but **not** in a statistically knowable direction
> (release-day return Welch *t* = 1.31, placebo *p* = 0.079), and the naive timer built on the
> uncertified point estimate doesn't even survive costs — payrolls Friday is **loud, not a
> tradable edge**.

## What we tested

We rebuild the "payrolls-day-effect" folklore on SPY daily total-return closes, 1997-01 →
2026-06, against the **actual** (not weekday-pattern-reconstructed) BLS Employment Situation
release calendar — 353 dates, the same source-verified table sibling study 602 already built
and cross-checked. The Signal axis splits release-day returns from the other 7,065 sessions
(Welch *t*, Newey-West *t*, Wilson hit rate, a two-sided 20,000-draw random-calendar
placebo), scans a [−3..+3] event window for a pre-release drift, and cross-checks the
realized SPY high-low range on the same days — the "is it just louder?" resolution test used
by [637-fomc-vol-crush](../637-fomc-vol-crush/). Tradability charges one-way costs × NAV on a
naive prior-close-entry / release-close-exit timer (one execution lag; the calendar is public
months ahead). **Dedup:** siblings [385-jobless-claims-momentum](../385-jobless-claims-momentum/)
(weekly initial-claims *momentum*, a different series and clock) and
[602-macro-announcement-premium](../602-macro-announcement-premium/) (the *pooled*
CPI+FOMC+NFP bundle, shown there to be an FOMC effect) never isolate the **NFP day itself** —
this study does. A 20-seed synthetic null plus a planted-effect world confirms the machinery
detects a comparable-sized effect without manufacturing one from noise. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "payrolls Friday" feels like a big deal, what actually happens to SPY that morning, and why loud isn't the same as predictable |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the Welch/HAC splits, the two-sided placebo, the event-window anatomy (and its multiple-comparisons caveat), the realized-range cross-check, the era contrast, the naive-timer cost sweep, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`payrolls_day_effect/`](payrolls_day_effect/). The NFP calendar is hardcoded from
actual BLS Employment Situation release dates (source-verified, shared with sibling study
602); SPY is an index-tracking ETF (no survivorship). **Not investment advice** — research &
education. See [LICENSE](../../LICENSE).*
