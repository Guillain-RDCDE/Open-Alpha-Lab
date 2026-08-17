# Study 950 — Zero-Coupon Convexity 📐

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the edge real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The theory's signs show up in **11 of the 12** fund × era × specification cuts (full census in [results](docs/results.md)), but the headline does not clear the bar: *b2* = **+110.5** (HAC *t* = **+0.84**; +154.7, *t* = +1.31 on realised variance), bootstrap CI [−163, +338], and the raw large-move bucket is **negative**. Two cuts *do* clear |*t*| = 2 (ZROZ 2010-2017, **+2.14** and **+2.88**) — one window of the cross-check fund, uncorrected for twelve looks, whose **next era flips both signs**. Significance that does not replicate next door, plus a significant **+0.72 yr** of residual duration (*t* = −2.73, both eras) meaning part of the spread is the 20s-30s curve, not convexity. |
| **Tradability** — is it bankable? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The duration-matched spread earns **−0.53%/yr** (monthly Sharpe −0.21, *t* = −0.87). On its own fitted terms convexity needs a **28 bp** monthly move to repay its carry cost; the median month delivers **15 bp**. It fails at the signal stage, not the friction stage — costs and the financing proxy never change the sign. |

> **In one sentence:** A zero-coupon Treasury fund really does carry ~42% more duration per dollar than the coupon long bond, and once you match that duration on the same rate factor the leftover payoff **leans the right way** in 11 of 12 cuts — a positive loading on the squared rate move, a negative intercept, exactly as "convexity is real but paid for" predicts — yet at ~7 bp for a 25 bp month it is an order of magnitude too small for eighteen years of tape to certify, the one window that does reach significance inverts in the window next to it, and what survives the match is as much a 20s-versus-30s curve trade as a convexity pickup.

## What we tested

Race **100% EDV** (20-30y STRIPS; **ZROZ** cross-check) against a **duration-matched
TLT + BIL mix** — the leverage `L ≈ 1.42` on TLT solved from each leg's rolling 252-day
realised beta to the **same rate factor**, the daily change in the 30-year yield (`^TYX`),
set at month end and traded the next month (one execution lag). Both arms excess-of-cash,
3 bps one-way on the mix's turnover, a **PROXY** 25 bp/yr financing spread on the levered
part (swept 0-100). The headline is not the average: it is the **asymmetry** —
`diff = a + b1·Δy + b2·Δy² `, run on both the squared net monthly move and realised
variance, plus move-size buckets, block-bootstrap CIs, three sweeps and a **full 12-cut
census** (2 funds × 3 eras × 2 specs — printed whole, so no claim rests on a subset).
EDV∩TLT∩BIL∩^TYX 2009-02-02 → 2026-06-30 (total-return closes, `auto_adjust=True`).
Survivorship is not a factor (named, still-listed funds) but **fund-selection** is: EDV and
ZROZ are the only two US zero-coupon Treasury ETFs long-lived enough to test.
**Dedup:** distinct from **884-convexity-barbell** (coupon barbell *vs* bullet, where the
barbell is the convex side and convexity is tested on the average), **826-treasury-duration-bab**
(a cross-sectional BAB sort, not a two-arm match), **924-cut-cycle-duration-extension**
(event-timed duration), and **380-curve-roll-down** / **864-yield-curve-twist** (first-order
curve effects — the `b1` this study hedges out and then reports when the hedge leaks).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what convexity is, why the zero fund should win the big months, what the tape actually paid, the 28 bp breakeven |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the rate-factor duration match, the asymmetry regression in both specs, bucket table, bootstrap CIs, era cut, sweeps, the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`zero_convexity/`](zero_convexity/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
