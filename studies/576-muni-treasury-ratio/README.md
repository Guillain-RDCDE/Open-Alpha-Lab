# Study 576 — Muni-Treasury-Ratio 🏛️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the M/T ratio time muni vs Treasury? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The sign is **right** (a high ratio → munis outperform) at *every* horizon, but the honest Newey-West *t* on the predictive slope is **+0.43** (h = 63) and tops out at **+1.28** (h = 252) — never the *t* ≥ 2 bar. The flashy quintile spread *t* of **+4.12** is an **overlap illusion**: on non-overlapping 63-day windows the same +0.74% spread carries a Welch *t* of just **+0.51**. Distribution-yield proxy, not the MMD curve. |
| **Tradability** — does the ratio-timer pay? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The Q5−Q1 forward-excess spread is **+0.60%** gross, **+0.35%** net (3 bps/leg + 50 bps/yr Treasury-leg borrow) — it clears costs numerically, but rests on a statistically insignificant, overlap-inflated edge (HAC *t* +0.43) measured on a coarse distribution-yield proxy rather than the tradable AAA-GO/MMD curve. Nothing bankable. |

> **In one sentence:** the muni-Treasury yield ratio times muni-vs-Treasury returns in the *right direction* — cheap munis do tend to outperform — but the effect never clears a robust *t* ≥ 2 on 17 years of the tape (HAC *t* +0.43, best-horizon +1.28), and the eye-catching quintile *t* of +4.12 is an overlap artifact that collapses to +0.51 on independent windows, on a distribution-yield proxy that is as much a tax artifact as a valuation signal.

## What we tested

The muni desk's oldest rich/cheap gauge: the **muni-Treasury yield ratio** (tax-exempt muni yield ÷
comparable Treasury yield). Because muni coupons dodge federal tax, the ratio normally sits below
1.0; the folklore says a *high* ratio means munis are *cheap* and should outperform, a *low* ratio
means *rich* and lag — a timing signal for muni or duration exposure. We proxy the ratio with the
trailing-12-month distribution yield of **MUB** over **IEF**, z-score it over a trailing year, and
test whether it predicts the forward muni-minus-Treasury excess return over 2009-2026 (4,275 daily
observations). The honest headline is a **Newey-West (HAC) predictive regression** (overlapping
forward windows are heavily autocorrelated), plus a quintile long-short with a two-sample *t*, a
**label-shuffle placebo**, a **non-overlapping** cross-check that exposes the overlap illusion,
costs + a short-leg borrow, a four-horizon robustness sweep, and a deterministic, seed-robust
synthetic positive control that plants a timing effect and proves the engine catches it. *Distinct
from the single-curve [132 Yield-Curve-Steepener](../132-yield-curve-steepener/) (Treasury slope
timer) — this is a **cross-sector relative-value ratio** and its muni-vs-Treasury excess.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the M/T ratio is, why "cheap munis outperform" sounds so sensible, and why the flashy edge evaporates once you count the windows honestly |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the HAC predictive regression, the quintile sort with its overlap-inflated *t*, the non-overlap kill, the placebo null, the four-horizon sweep, costs + borrow, and the seed-robust synthetic positive control |

The fingerprinted real-data run (MUB/IEF, 2009-06 → 2026-06, tape fp `738df551786b`, analysis fp
`c3e43ab615e6`, as-of 2026-06-30) is in [docs/results.md](docs/results.md); the offline machinery
proof runs on the deterministic synthetic world in
[`muni_treasury_ratio/data.py`](muni_treasury_ratio/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`muni_treasury_ratio/`](muni_treasury_ratio/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
