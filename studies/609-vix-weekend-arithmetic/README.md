# Study 609 — VIX-Weekend-Arithmetic 📅

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there a calendar-arithmetic seesaw in ^VIX? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | Enormous and mechanical: Monday **+1.81 %/day**, Friday **−0.98 %/day**, Mon−Fri spread **+2.78 %/day** at Welch *t* = **11.6** / Newey-West *t* = **11.2**, placebo **p < 1/20,000**, *t* ≥ 3.3 in **every decade** since 1990, and it follows calendar gaps (holidays included) at *t* = 13.2. Direction exactly as the variance-day-count arithmetic predicts — and **opposite to the folk retelling** ("up into Friday" is backwards). Magnitude ≈ 40 % of the full bound: the market prices a weekend day at ~**60 %** of a trading day's variance (implied *f* = 0.597). |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The index is **not tradable**, and the pop never reaches the vehicle: while ^VIX jumps **+1.63 %** on the average Monday, VIXY *loses* **−0.32 %** over the same weekend. The literal buy-Friday-close / sell-Monday-close harvest bleeds **−15.2 %/yr gross** and **−19.9 %/yr net** at 5 bps. Futures price the forward VIX — the calendar is already in there. |
| **"Can a VIX ETP harvest the Monday pop?"** | ![Busted](https://img.shields.io/badge/ETP_harvest%3F-Busted-8b949e?style=flat-square) | 728 weekends over 15.5 years: VIXY's Monday-vs-rest Welch *t* = **−0.92** with a **negative** mean. The seesaw is a property of the index's *formula*, fully anticipated in every instrument you can actually buy. |

> **In one sentence:** the VIX really does have a guaranteed weekend seesaw baked into its own
> 30-calendar-day formula — **down** into Friday as the weekend enters the window, a **+1.8 %**
> pop on Monday once it has passed (HAC *t* ≈ 11, every decade, holidays too; the popular
> "up-into-Friday" version has the sign backwards) — running at ~40 % of the full arithmetic
> because option markets price a weekend day at ~60 % of a trading day's variance; but it lives
> only in the index's formula: VIX futures anticipate it, so the tradable weekend hold *loses*
> ~20 %/yr net — Real, and a perfect Mirage.

## What we tested

The VIX squares to expected S&P-500 variance over the next **30 calendar days**, annualized on
a calendar clock — but variance accrues mostly on **trading days**. A window quoted on Friday
holds 20 trading + 10 weekend days; quoted on Monday, 22 + 8. Pure arithmetic then forces the
quote down into the weekend and up after it. On the full ^VIX tape (1990 → 2026-06-30, 9,190
daily changes) we build the day-of-week table of Δln VIX, test the Monday-vs-Friday contrast
with a Welch *t*, a Newey-West(10) dummy-regression *t* and a 20,000-draw label-shuffle
placebo, then race the magnitude against the one-parameter day-count model to extract the
market's implied weekend variance fraction (*f* = 0.597). Robustness: by decade and by
*calendar gap* (post-holiday days behave like Mondays). The third axis asks whether a tradable
VIX-futures ETP (VIXY, total-return, 2011+) inherits the pop: buy Friday close, sell Monday
close, one round trip per weekend at 2/5/10 bps one-way — the calendar-known-in-advance entry
is the one documented execution lag. A deterministic synthetic control (log-AR(1) vol quoted
through the arithmetic with a planted *f*) proves the machinery recovers a planted seesaw and
stays silent on the null. Siblings: [90-weekend](../90-weekend/) is the *equity* weekend
effect; [375-vxx-roll-decay](../375-vxx-roll-decay/) is the *futures-carry* bleed — this study
is the index's **calendar arithmetic** itself. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the fear gauge falls on Fridays and jumps on Mondays by construction, how big the seesaw really is, why the popular version gets the sign backwards, and why you still can't make a cent off it — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the day-of-week table with HAC inference, the label-shuffle placebo, the day-count model race and the implied weekend fraction, decade + calendar-gap robustness, the VIXY leak test with costs, and the planted-*f* synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`vix_weekend_arithmetic/`](vix_weekend_arithmetic/). The signal is the weekday of the
close (known in advance forever); the third axis charges realistic ETP costs on the literal
weekend hold. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
