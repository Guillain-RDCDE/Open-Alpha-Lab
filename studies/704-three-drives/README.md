# Study 704 — Three Drives 🎯

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the third drive reverse? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Across **110** algorithmically-detected Three-Drives patterns (SPY + a 7-ticker basket, 1993→2026), fading the third drive nets **+45.0 bps/trade** at *t* = **+0.62** — nowhere near the desk's *t* ≥ 2 bar — and sits inside a coin-flip (random time, random direction) placebo cloud (*p* = 0.21). Flat at every hold period (5/10/20/40 days, |*t*| < 2 throughout) and across every ZigZag threshold (3%-8%). A synthetic control proves the harness *would* catch a real planted reversal (*t* up to +6.9), so this is a genuine null. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | There is no edge to charge costs against, and detections are thin — 110 across 8 broad tapes over 20-30+ years, a handful a year pooled across the whole basket. Nothing to deploy. |
| **"Are the Fibonacci ratios load-bearing?"** | ![Busted](https://img.shields.io/badge/Fibonacci_load--bearing%3F-Busted-8b949e?style=flat-square) | Swap the specific 0.382-0.886 / 1.13-2.618 targets for a **random** ratio grid, keep every other rule: the real grid doesn't clear significance (*p* = 0.26). The pattern's own word — "symmetric" — fares no better: patterns whose three drives are more evenly spaced in *time* show only a directionally positive, uncertified lean (Welch *t* = +1.39, n = 110). |

> **In one sentence:** three Fibonacci-proportioned pushes to a new high or low — each correction
> retracing 0.382-0.886 of the drive before it, each drive extending the prior correction by
> 1.13-2.618× — get algorithmically detected **110** times across SPY and a 7-ticker basket, and
> fading the reversal at the third drive earns **+45 bps** at *t* = **0.62**, indistinguishable
> from a coin flip (*p* = 0.21): a clean **None × Mirage**, and neither the specific Fibonacci
> ratios nor the folklore's own "symmetric" claim survives an honest placebo (**Busted**).

## What we tested

The "Three Drives" pattern (Larry Pesavento's harmonic-trading lineage, taught alongside Elliott
Wave and Gartley material) reads five labelled swing pivots — point 1 through point 5, with an
implicit start "point 0" — forming **three drives** to a new high/low, separated by **two
corrections**, each leg Fibonacci-proportioned to the one before it: "three symmetric pushes"
that supposedly exhaust the trend and reverse hard once drive 3 completes. We detect it
mechanically with a **4% ZigZag** on SPY + a 7-ticker broad-index/ETF basket: six consecutive
alternating pivots qualify if both corrections retrace 0.382-0.886 of the drive before them, both
drives extend the prior correction by 1.13-2.618×, and each drive genuinely extends beyond the
last (the "extending drives" rule every source agrees on) — the widest mechanical band a
proponent would accept. Unlike Wolfe Waves' EPA line or Gartley's D-point-plus-symmetry, Three
Drives makes **no price-target claim** — just a plain reversal — so we test the one thing it does
claim: fade the pattern at the next close after point 5's confirmation (one documented execution
lag, no look-ahead) against fixed 5/10/20/40-day horizons, benchmarked against the honest base
rate for a ±1-direction bet — a **random-time, random-direction coin flip**, not a drift-matched
baseline. A Fibonacci-grid placebo (random ratio targets, same machinery) and a time-symmetry
myth-check (does "symmetric" itself predict anything?) probe whether the specific numbers matter.
A deterministic synthetic control with a planted-reversal knob proves the harness is live.
**Dedup:** [445-elliott-wave](../445-elliott-wave/) (a different wave count and claim),
[697-wolfe-waves](../697-wolfe-waves/) (the closest cousin in shape — a converging wedge with a
price-and-time EPA target, not three Fibonacci-proportioned pushes), and
[468-gartley-harmonic](../468-gartley-harmonic/) (the same Fibonacci-grid-placebo idiom, one
drive/correction cycle instead of three). As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a Three Drives pattern is supposed to look like, why the Fibonacci story is seductive, and what actually happens when a computer draws the wedges instead of a chartist with hindsight |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the ZigZag/geometry detector, the fade-vs-coin-flip test, the Fibonacci ratio-grid placebo, the time-symmetry split, the threshold sweep, and the synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`three_drives/`](three_drives/). SPY + basket are broad indices/ETFs (no cross-sectional
survivorship on the Signal axis). **Not investment advice** — research & education. See
[LICENSE](../../LICENSE).*
