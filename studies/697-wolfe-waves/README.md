# Study 697 — Wolfe Waves 🌀

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the EPA target get hit? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Across **370** algorithmically-detected Wolfe waves (SPY + a 7-ticker basket, 1993→2026), the "Estimated Price at Arrival" target is hit **34.8%** of the time vs **34.6%** for a same-distance target placed on a **random day** (*z* = +0.05, *p* = 0.47). Flat across the ZigZag threshold (36.0%–37.8%, no trend) and a geometry-free fixed-horizon cut (|*t*| < 2 at every horizon, 5/10/20/40 days). A synthetic control proves the harness *would* catch a real planted reversal (*t* up to 5.6), so this is a genuine null. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | There is no edge to charge costs against — the target-hit rate matches chance and the directional returns are flat-to-mildly-negative at every horizon. Nothing to trade. |
| **"No reliable timer"?** | ![Confirmed](https://img.shields.io/badge/No_reliable_timer%3F-Confirmed-8b949e?style=flat-square) | The pattern's own price-*and-time* claim fails on the time axis too: the projected time target correlates **+0.04** with when a winning trade actually resolves (mean absolute error **40.8 bars**, on trades that average 21.7 bars to resolution). Even the rare wins the target does hit, the clock is noise. |

> **In one sentence:** Bill Wolfe's five-point "natural equilibrium" wedge — an algorithmically
> detected 1-2-3-4-5 structure whose point-5 throw-over is supposed to project a precise
> price-and-time target via the EPA line — gets its target hit **34.8%** of the time, statistically
> identical to a **same-distance random target on a random day** (*z* = 0.05), and its time
> projection correlates **+0.04** with reality: a **None x Mirage**, and the "time" half of the
> claim is separately **Busted**.

## What we tested

Wolfe Waves claim a converging 5-point wedge (points 1-2-3-4-5, alternating swing pivots) whose
final leg overshoots the trendline through points 1 and 3 — the "throw-over" — right before price
snaps back toward the **EPA line** (the trendline through points 1 and 4, extended forward), on
both price *and* time. Wolfe's own rules were never published as a single canonical checklist —
every trading-education source states them slightly differently, itself evidence of how much
discretion the pattern leaves a human chartist — so we encode the **one rule everybody agrees on**
(point 5 pierces the 1-3 line) plus the channel/convergence checks most sources add, detected
mechanically by a **4% ZigZag** on SPY + a 7-ticker broad-index/ETF basket. We test the one
falsifiable claim: does the EPA price target get hit before the invalidation stop **more often
than a same-distance target on a random day** (the same idiom sibling
[448-point-and-figure](../448-point-and-figure/) uses for its count target) — and does the EPA
*time* target predict *when*. One documented execution lag (enter the close one bar after point
5's ZigZag confirmation — no look-ahead); a ZigZag-threshold sweep; a geometry-free fixed-horizon
robustness cut; a deterministic synthetic control with a planted-reversal knob. **Dedup:**
[445-elliott-wave](../445-elliott-wave/) (a different wave count, Fibonacci rules, no price
target), [447-gann-angles](../447-gann-angles/) (a single trend line, no 5-point structure),
[448-point-and-figure](../448-point-and-figure/) (the closest cousin in *method* — same
target-hit-vs-random-day idiom, opposite verdict) and [704-three-drives](../704-three-drives/)
(a different Fibonacci five-point structure, no EPA line). As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a Wolfe Wave is supposed to look like, why the "natural equilibrium" story is seductive, and what actually happens when you let a computer draw the wedges instead of a chartist with hindsight |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the ZigZag/geometry detector, the target-hit test vs the random-day placebo, the time-target correlation, the threshold sweep, the fixed-horizon HAC cut, and the synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`wolfe_waves/`](wolfe_waves/). SPY + basket are broad indices/ETFs (no cross-sectional
survivorship on the Signal axis). **Not investment advice** — research & education. See
[LICENSE](../../LICENSE).*
