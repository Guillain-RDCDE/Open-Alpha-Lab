# Results — Study 586 (Liquidation-Cascade): the crypto capitulation-bottom folklore

*Generated from [`liquidation_cascade/`](../liquidation_cascade/) over this study's **deterministic
synthetic** tape — a 1,500-day BTC-style series with an explicit forced-liquidation channel
(``synthetic_series(bounce_alpha=0.0, horizon=5, seed=586)``; frame fingerprint `b3d7960bf851`).
There is **no free real forced-liquidation feed**, so the headline run is the synthetic **null**
world; a real Coinglass/Amberdata tape staged at ``_cache/liquidation_series.parquet`` would flow
through the identical engine. As-of **2026-06-30**.*

## The verdict, earned — Signal `NONE` · Tradability `MIRAGE`

The folklore: a big wave of **forced liquidations** (over-leveraged longs margin-called and dumped
by the exchange) marks a *capitulation bottom* — the mechanical selling overshoots and crypto
**bounces**, so "buy after the liquidation spike" should pay. We run it as an **event study**:
forward returns after large-liquidation days vs the unconditional baseline, a two-sample *t*, a
label-shuffle placebo, a horizon × threshold robustness sweep, costs, and a seed-robust synthetic
positive control.

Two things decide the stamps. **(1) There is no free real liquidation tape** — the per-day
aggregate forced-liquidation USD series the claim needs (Coinglass/Amberdata/exchange futures
APIs) is paywalled with no usable free history, so this study is **synthetic-only** and *cannot*
earn `REAL` (that requires a robust *t* ≥ 2 on a **real** tape). **(2) On the synthetic null world**
— where liquidations carry no forward information by construction — the engine correctly finds
**no bounce**: the 5-day forward return after a top-5%-liquidation day is **−1.20%** versus a
baseline of **−0.29%**, a gap of **−0.96%** (two-sample *t* **−0.92**, placebo *p* **0.34**), and
the gap **flips sign across horizons and thresholds** with no |t| ≥ 2 anywhere. So `NONE` on
signal (no real tape; the null world shows the detector doesn't hallucinate a bounce), `MIRAGE` on
tradability (an unreachable data feed, a sign-unstable gap, a 20 bps round-trip that turns even the
best synthetic cut net-negative). The synthetic positive control proves the engine *does* catch a
planted bounce (mean event-*t* climbs from **+0.13** at the null through **+2.89** to **+5.24** as
the bounce is planted stronger) — so this is an honest null, not a broken engine.

## Data stamp

- **Synthetic tape**: 1,500 daily rows (2019-01-01 →), columns ``ret``, ``liq``, ``price``;
  ``bounce_alpha = 0`` (null), ``horizon = 5``, ``seed = 586``; frame (``ret``, ``liq``)
  fingerprint `b3d7960bf851`
- **Event days** (top 5% of trailing 252-day liquidation flow): **74 of 1,500** (4.93%)
- **Real tape**: none — ``fetch_series(fetch=False)`` returns an empty frame; there is no free
  forced-liquidation feed (the data-availability limitation, named on the SIGNAL axis)

## The event study — no bounce in the null world

| | value |
|---|--:|
| Forward 5-day return after a liquidation event (74 days) | **−1.20%** |
| Baseline forward 5-day return (all days) | **−0.29%** |
| Gap (event − baseline) | **−0.96%** |
| Two-sample (Welch) *t* | **−0.92** |
| Label-shuffle placebo *p* (2,000 perms) | **0.34** |

The folklore predicts a *positive* gap (a bounce). The null world delivers a small *negative*
gap that the placebo cannot distinguish from noise (*p* = 0.34) — exactly what an honest detector
should print when there is nothing there.

## Robustness — the gap has no stable sign

Event-minus-baseline gap (%) and its *t*, across horizons × event thresholds (null world):

| horizon | q=0.90 gap% (t) | q=0.95 gap% (t) | q=0.99 gap% (t) |
|--:|--:|--:|--:|
| 1  | −0.40 (−1.06) | −0.64 (−1.30) | +0.21 (+0.19) |
| 3  | −0.35 (−0.57) | +0.09 (+0.12) | +0.96 (+0.61) |
| 5  | −1.10 (−1.37) | −0.96 (−0.92) | +2.92 (+1.27) |
| 10 | −1.45 (−1.38) | −0.70 (−0.55) | −0.32 (−0.12) |
| 20 | −1.69 (−1.09) | −0.94 (−0.43) | −1.48 (−0.39) |

Every cell is |*t*| < 2, and the sign flips between cells (the biggest positive, +2.92% at h=5/
q=0.99, rests on only 15 event days and *t* +1.27). No stable, significant post-liquidation bounce
anywhere in the grid.

## Costs

| | value |
|---|--:|
| Best synthetic gap (h=5, q=0.99, 15 events) | **+2.92%** |
| Net of a 20 bps round-trip (10 bps/side × NAV) | **+2.72%** |
| Headline gap (h=5, q=0.95) | **−0.96%** |
| Net (20 bps round-trip) | **−1.16%** |

Costs are a footnote: the headline gap is negative before you pay to trade it, and the one
positive cut survives costs only because it is a 15-event, *t* = 1.27 artefact — not a signal.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

Mean event-minus-baseline *t* over 25 synthetic worlds (h=5, q=0.95), as the bounce is planted
progressively stronger:

| Planted `bounce_alpha` | Mean event-*t* (25 seeds) | reads as |
|--:|--:|---|
| 0.000 (null) | **+0.13** | flat — no false signal |
| 0.005 | +1.05 | bounce emerging |
| 0.010 | +1.98 | at the bar |
| 0.015 | **+2.89** | clears *t* = 2 |
| 0.020 | +3.74 | clear bounce |
| 0.030 | +5.24 | strong bounce |

At the null the event-*t* is ≈ 0 (no hallucinated bounce); planting a genuine capitulation-bounce
(`bounce_alpha > 0`) drives the event-*t* positive and past +2 as it grows. The detector works — so
the null-world result is a statement about *a world where liquidations carry no forward
information*, not a broken engine. (Control only; never cited for a real-tape stamp — there is no
real tape.)

## Why the folklore can't certify here

1. **No free data.** The claim is about *forced-liquidation flow*, and the per-exchange aggregate
   liquidation-USD series it needs is a paid product (Coinglass, Amberdata) with no usable free
   history. A no-key retail stack cannot build the real event study — so the honest ceiling is
   `WEAK`/`NONE`, and we publish the synthetic machinery + null instead of faking a real number.
2. **The contemporaneous crash is not the signal.** Liquidations spike *with* the down day
   (mechanically), so "liquidations ⇒ price fell" is a tautology. The tradable claim is the
   *forward bounce*, which the event study isolates by looking only at days *after* the event — and
   in the null world there is nothing there.
3. **Even if real, sign-instability is fatal.** In the null world the gap already flips sign across
   horizons and thresholds. A real bounce would have to survive that same grid with a stable sign
   and |t| ≥ 2 — a high bar the folklore has never cleared on a public tape.

## The honest takeaway

"Buy the blood after a liquidation cascade" is a great story, but it needs a forced-liquidation
tape nobody gives away for free — so on a no-key stack it is **untestable on real data**, which
caps it at `NONE`. On the deterministic synthetic **null** world the engine finds **no bounce**
(gap −0.96%, *t* −0.92, placebo *p* 0.34) and a sign that flips across the horizon × threshold
grid; the positive control confirms the same engine banks a planted bounce past *t* = 2. `NONE` ×
`MIRAGE`: a plausible microstructure story with no free way to test it and no bounce in the null.
