# Results — Study 571 (Pension-Underfunding): the Franzoni-Marin funding-status anomaly

*Generated from [`pension_underfunding/`](../pension_underfunding/). **This study is
synthetic-only** — the point-in-time pension-footnote data the real anomaly needs (projected
benefit obligation, plan assets, funded status scaled by market cap) is not reachable from a
no-key retail stack, so there is **no real tape** here. The headline runs on the deterministic
synthetic cross-section (`underfunding_alpha = -0.05`, `idio_vol = 0.18`, seed **571**, 300
firms), panel fingerprint `b43234ffbd01`. As-of **2026-06-30**.*

## The verdict, earned — Signal `NONE` · Tradability `MIRAGE`

Franzoni & Marin (2006) document the **pension-underfunding anomaly**: firms whose defined-benefit
pension plans are most underfunded — the biggest pension hole relative to equity — go on to earn
*anomalously low* returns (roughly −7%/yr in their sample), as if the market under-reacts to this
senior, off-balance-sheet liability. We build the funding measure `funding_gap = (assets −
PBO)/mktcap`, sort firms by the depth of the hole, and test whether well-funded names beat
underfunded ones.

**The signal is capped at `NONE` for one honest reason: there is no real tape.** The real
pension-footnote panel is not free (see the data caveat below), so we cannot clear the desk's
inference bar — *a robust t ≥ 2 on a **real** tape*. What we *can* show is that the engine is
faithful: on the deterministic synthetic world with a planted Franzoni-Marin effect it recovers
the puzzle cleanly (firm-level slope-*t* **−4.64**, IC **−0.26**, quintile long-short spread
**+8.79%/yr** at *t* **2.45**, placebo *p* **0.006**), and it stays flat at the null (mean
slope-*t* **+0.01**, IC **+0.001** over 25 seeds). So this study is a *machinery proof plus a data
caveat*, not a claim on the tape: `NONE` on Signal (no real evidence), `MIRAGE` on Tradability (a
short leg of stressed old-economy names, and no free way to even source the signal).

## Data caveat — why this is synthetic-only (named on the SIGNAL axis)

The Franzoni-Marin funding measure needs **point-in-time pension-footnote data**: the projected
benefit obligation (PBO), plan assets, or the post-SFAS-158 balance-sheet net funded status,
scaled by market cap. That lives in the 10-K pension footnote (Compustat pension items). A no-key
retail stack (yfinance) exposes **none** of it — `.balance_sheet` carries no pension line, and
there is no free point-in-time pension-footnote panel. Reconstructing the anomaly honestly needs
Compustat. So [`data.py`](../pension_underfunding/data.py)'s `real_panel()` is a documented stub
that returns an empty frame, `HAVE_REAL` is always `False`, and the Signal axis is capped at
`NONE`. A synthetic-only study can *never* be `REAL`.

## The synthetic sort — the engine catches the planted puzzle

At the planted strength (`underfunding_alpha = -0.05`, a realistic ~-5%/unit tilt matching the
Franzoni-Marin magnitude):

| Quintile bucket (60 firms) | Forward return |
|---|---|
| **Well-funded** (smallest hole) | **+7.15%** |
| **Underfunded** (deepest hole) | **−1.64%** |
| **Spread (well − underfunded)** | **+8.79%** (two-sample *t* **2.45**, placebo *p* **0.006**) |

The puzzle predicts well-funded > underfunded (a *positive* spread), and the engine recovers it:
the well-funded quintile beats the deeply-underfunded one by ~8.8pp. The label-shuffle placebo
*p* = 0.006 confirms this is not a bucketing artefact **in the synthetic world**.

## The firm-level relation — the sign IS the puzzle

| | value |
|---|---|
| Slope (forward_ret on hole depth) | **−4.69%** per depth unit |
| Slope *t* | **−4.64** (a *negative* slope is the puzzle) |
| Information coefficient (IC) | **−0.26** |

A *negative* slope/IC is the Franzoni-Marin puzzle: the deeper the pension hole, the lower the
forward return — recovered cleanly by the engine on the planted world.

## Robustness — the sign survives design choices (in the synthetic world)

| Design | Spread (well − underfunded) | Long-short *t* | Firm slope-*t* |
|---|---|---|---|
| Quintile (20%), equal-weight | **+8.79%** | 2.45 | −4.64 |
| Decile (10%), equal-weight | **+14.86%** | 3.14 | −4.64 |
| Tercile (33%), equal-weight | **+9.14%** | 3.49 | −4.64 |
| Quintile (20%), cap-weight | **+5.93%** | 2.45 | −4.64 |

The sign is stable across bucket definitions and weighting; the spread is a touch smaller
cap-weighted (mega-caps carry smaller pension holes), as expected.

## Costs

| | value |
|---|---|
| Gross spread (well − underfunded, quintile) | **+8.79%** |
| Net (5 bps/leg round-trip + 150 bps/yr borrow, 1y hold) | **+7.09%** |

Even in the synthetic world where the effect *exists*, the underfunded short leg — stressed
old-economy names — carries a punitive 150 bps borrow, eating ~1.7pp. On a real tape (which we do
not have) that short leg is exactly the hard-to-borrow tail.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `underfunding_alpha` | Mean slope-*t* | Mean IC | Mean spread | Mean long-short *t* |
|---|---|---|---|---|
| 0.00 (null) | **+0.01** | **+0.001** | −0.04% | −0.04 |
| −0.02 | −1.91 | −0.109 | +4.92% | 1.47 |
| −0.05 (headline) | **−4.79** | −0.266 | +12.36% | 3.67 |
| −0.08 | −7.67 | −0.404 | +19.80% | 5.71 |

At the null the slope-*t* and IC are ≈ 0 — no false signal. Planting a genuine underfunding effect
(`underfunding_alpha < 0`) drives the slope negative and past −2 as it grows. The detector works —
so the *absence* of a real-tape verdict here is a **data limitation, not a broken engine**. (Control
only; never cited as a real-tape result.)

## The honest takeaway

The pension-underfunding anomaly is real in the literature (Franzoni-Marin 2006; the market
under-reacts to off-balance-sheet pension leverage). This study proves the *machinery* would catch
it — a clean negative slope, a ~+8.8%/yr well-minus-underfunded spread, a flat null. But because the
point-in-time pension-footnote data is not free, **there is no real tape to certify it on**, so the
Signal axis is honestly `NONE` (not evidenced on a real tape, not `REAL`) and Tradability is
`MIRAGE` (a costly-to-borrow short leg you cannot even source cheaply). The literature support keeps
this from being a busted claim — it is a real anomaly the retail stack simply cannot reach.
