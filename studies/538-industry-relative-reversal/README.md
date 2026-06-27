# Study 538 -- Industry-Relative-Reversal

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Industry-relative reversal does what Hameed-Mian (2015) say: it **flips the raw signal's sign and beats it** (IRR **+9.5** vs RAW **-11.0** bps/mo), survives a one-month gap where the raw bid-ask-bounce reversal dies (skip=1 *t* = **+1.33**), and the industry adjustment is decisively real on a label-shuffle placebo (p = **0.005**). But the level is tiny: IRR HAC *t* = **+0.56**, no sub-period clears 2.0. Right mechanism, sub-bar magnitude. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | ~**77%** one-way monthly turnover; **break-even ~3.1 bps**; net spread negative by 5 bps and a significant loss (*t* = -3.32) by 20 bps. A sub-2%/yr gross edge at 100% monthly refresh cannot be traded. |
| **Does industry-adjustment beat raw?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | On the real tape (IRR +9.5 vs RAW -11.0 bps/mo; IRR survives skip=1, RAW does not) **and** the synthetic control (IRR *t* = **69** vs RAW *t* = **31** at the same planted within-industry reversal), the within-industry sort is the cleaner signal. |

> **In one sentence:** netting out the industry move really does isolate a cleaner one-month reversal than the raw Jegadeesh sort -- it flips the sign, beats it, survives the bid-ask-bounce gap, and the industry adjustment is provably real (placebo p = 0.005) -- but on a 54-name survivor basket the residual edge is a whisper (*t* = +0.56), and at ~77% monthly turnover with a ~3 bps break-even it is untradable.

## What we tested

Hameed-Mian (2015) and Da-Liu-Schaumburg (2014) argue the short-term reversal is much
stronger when measured **industry-relative**: decompose last month's return into an
*across-industry* part (the sector's own move, which does **not** reverse) and a
*within-industry* part (the stock minus its sector, which does). We build the industry-
adjusted one-month reversal on a fixed 54-name, six-GICS-sector S&P 500 survivor basket
(1990-2026) and run it head-to-head against the **raw** one-month reversal of
[Study 329](../329-one-month-reversal/). The teardown pins it against four things: a
**placebo** that shuffles the industry labels (does "industry" mean anything?), a
one-month-gap variant (bid-ask bounce), a sub-period split, and a realistic turnover/cost
sweep with short borrow. A deterministic synthetic panel -- a non-reversing industry factor
plus a planted within-industry reversal -- proves the engine isolates the within component;
the verdict is measured on the market.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "subtract the sector first" gives a *better* loser-minus-winner sort than the raw version -- and why a better signal can still be a worse-than-nothing trade |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | RAW vs IRR HAC *t*, the industry-label-shuffle placebo (p = 0.005), the skip=1 bounce test, sub-period decay, the cost wall & break-even, and the within-industry synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`industry_relative_reversal/`](industry_relative_reversal/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
