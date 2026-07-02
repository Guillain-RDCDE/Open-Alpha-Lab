# Results — Study 574 (Penny-Beat): the earnings-management discontinuity

*Generated from [`penny_beat/`](../penny_beat/) over the **deterministic synthetic** firm-quarter
panel (seed 574, n = 6000; `spike = 0.55`, `penny_penalty = −0.05`). Panel fingerprint
`1ad98df82652`; surprise-histogram fingerprint `3b43e570d8d5`. As-of **2026-06-30**. There is **no
real tape**: a survivorship-free, point-in-time consensus-vs-actual EPS panel is a licensed I/B/E/S
product a no-key retail stack cannot reach — so this study is synthetic-only and its Signal ceiling
is `WEAK` (a `REAL` stamp needs a robust t on a real tape).*

## The verdict, earned — Signal `WEAK` · Tradability `MIRAGE` · "Discontinuity real?" `CONFIRMED`

The claim (Burgstahler & Dichev 1997; Degeorge, Patel & Zeckhauser 1999): firms that *just barely*
beat consensus EPS — by exactly ~1 cent — look **managed**. The tell is a **discontinuity** in the
earnings-surprise histogram (a deficit of small misses, an excess spike at +$0.01), and the trading
claim on top is that these penny-beaters go on to earn **weaker** returns than decisive beaters.

On the synthetic tape the engine cleanly **detects the discontinuity** (z = **+17.7**, placebo
*p* = **0.0005**) and finds a penny-minus-decisive return spread of **−4.9pp** (two-sample *t*
**−9.2**, placebo *p* = **0.0005**). But the honest decomposition shows that spread is **not all
management**: with the return penalty knob set to *zero* (`penny_penalty = 0`, spike still on) the
penny bucket *still* underperforms decisive beaters by **−1.8pp** (*t* **−3.5**) — a mechanical
**PEAD composition** gap, because a +1c surprise is a *smaller* surprise than a ≥+3c decisive beat
and so drifts less regardless of any manipulation. Netting composition out (managed vs honest firms
*within* the +1c bin) leaves a clean planted penalty of **−5.8pp** (*t* **−6.7**). So the effect is
real *by construction here*, the discontinuity is `CONFIRMED` (in the literature and the synthetic),
but the **Signal is `WEAK`**: no real tape, and the naive penny-minus-decisive spread over-states
the management penalty by conflating it with PEAD.

## Data stamp

- **Synthetic panel**: 6000 firm-quarters, seed 574, `spike = 0.55`, `penny_penalty = −0.05`,
  surprise sd = 6c; fingerprint `1ad98df82652`
- **Surprise histogram** (−12c..+12c): fingerprint `3b43e570d8d5`

## The discontinuity — the +$0.01 spike and the just-below-zero deficit

Surprise-histogram counts (cents), a smooth distribution would have **no** step at +1c:

| Surprise (¢) | −3 | −2 | −1 | 0 | **+1** | +2 | +3 | +4 |
|---|---|---|---|---|---|---|---|---|
| Count | 151 | 164 | 180 | 400 | **988** | 363 | 362 | 301 |

| | value |
|---|---|
| +1c bin count | **988** |
| neighbour average (0c, +2c) | 381.5 |
| excess mass at +1c | **+606** |
| **discontinuity z** | **+17.7** |
| discontinuity placebo *p* | **0.0005** |
| just-below-zero deficit ratio (misses −1..−3 / beats +2..+4) | **0.48** |

The +1c bin holds **988** firm-quarters against a smooth expectation of ~382 — an excess of **606**
(z +17.7). The small-miss bins (−1..−3c) are *depleted* to 48% of the mirror small-beat bins: mass
was moved *up* across zero. That is the earnings-management fingerprint.

## The return test — penny-beaters vs decisive beaters

| Bucket | Forward return | n |
|---|---|---|
| **Penny-beaters** (surprise = +1c) | **−1.07%** | 988 |
| **Decisive beaters** (surprise ≥ +3c) | **+3.84%** | 2067 |
| **Spread (penny − decisive)** | **−4.92%** (two-sample *t* **−9.22**, placebo *p* 0.0005) | |

The claim's direction (penny < decisive) holds. But see the decomposition — half of this is not
management.

## The honest decomposition — composition vs management

| | spread (penny − decisive) | *t* | reads as |
|---|---|---|---|
| Full panel (`penny_penalty = −0.05`) | **−4.92%** | **−9.22** | management **+** PEAD composition |
| Composition-only (`penny_penalty = 0`, spike on) | **−1.83%** | **−3.48** | pure PEAD (smaller surprise drifts less) |
| Within-+1c-bin, managed vs honest (clean penalty) | **−5.76%** | **−6.69** | the planted management penalty, composition netted out |

A +1c surprise is a *smaller* honest surprise than a ≥+3c decisive beat, so it drifts less under a
mild post-earnings-announcement-drift slope **even with zero manipulation** — the −1.8pp
composition gap. The naive penny-minus-decisive spread therefore *over-states* the management
penalty. The within-bin test (managed vs honest firms that *both* landed on +1c) isolates the clean
planted penalty (−5.8pp) — but it needs the *latent* "managed" label, which a real analyst never
sees. This confound is exactly why the naive spread reads `WEAK`, not `REAL`, even before the
missing-real-tape problem.

## Robustness — the sign is stable across earnings-management intensities

| Spike (fraction of small-misses managed up) | Discontinuity z | Penny − decisive spread | *t* |
|---|---|---|---|
| 0.20 | +8.0 | −3.3% | −5.23 |
| 0.40 | +13.9 | −4.3% | −7.52 |
| 0.55 (headline) | +17.7 | −4.9% | −9.22 |
| 0.70 | +21.3 | −5.2% | −10.30 |

The discontinuity z and the (negative) return spread both grow monotonically with the management
intensity — the sign never flips. Stable *given the planted world*; that stability is a statement
about the engine, not a real tape.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `penny_penalty` (spike = 0.55) | Mean discontinuity z (25 seeds) | Mean penny−decisive *t* (25 seeds) |
|---|---|---|
| 0.00 (return null) | **+17.9** | −3.7 (PEAD composition floor) |
| −0.02 | +17.9 | −6.0 |
| −0.05 (headline) | +17.9 | −9.3 |
| −0.10 | +17.9 | −14.5 |
| **joint null** (spike = 0, penalty = 0) | **+0.3** | **−2.3** |

Reading: the **discontinuity z is ≈ 0 at the joint null** (spike off) and jumps to ~18 the moment a
spike is planted — the discontinuity detector is faithful, false-signal-free at the null. The
penny−decisive *t* falls monotonically as the planted penalty deepens, but even at the *return*
null (penalty 0) it sits at ≈ −3.7 because of the PEAD composition floor described above — the
detector recovers a planted penalty, but the naive spread carries a mechanical head-start that a
publication-grade test must net out. (Control only; never cited for a real-tape stamp — there is
no real tape.)

## Costs — the tradable expression

The trade would be **short penny-beaters, long decisive beaters** (harvest the negative spread).

| | value |
|---|---|
| Gross penny − decisive spread | **−4.92%** |
| Harvestable magnitude, net (5 bps/leg round-trip + 100 bps/yr borrow, 0.25y hold) | **+4.47%** |

Costs are a footnote against the raw synthetic magnitude — but the *tradability* problem is not
costs, it is (a) no reachable data, (b) a tiny per-quarter penny bucket, (c) shorting exactly the
firms you'd expect to be crowded, and (d) half the "edge" being mechanical PEAD you could harvest
more directly. `MIRAGE`.

## Why the claim doesn't certify here

1. **No real tape.** The consensus-vs-actual EPS panel is licensed (I/B/E/S); a no-key stack cannot
   build the histogram on real data. Synthetic-only ⇒ Signal ceiling `WEAK`.
2. **The return penalty is confounded with PEAD.** The naive penny-minus-decisive spread over-states
   the management penalty because a penny beat is a *smaller* surprise; only a within-bin (managed
   vs honest) test isolates the clean effect, and that needs a label no real analyst can see.
3. **The discontinuity is the robust part.** The +$0.01 spike / just-below-zero deficit is
   well-established in the literature and cleanly detected here — but a *shape* in a histogram is a
   diagnostic of aggregate behaviour, not a per-firm tradable return signal.

## The honest takeaway

The penny-beat *discontinuity* is real and famous (Burgstahler-Dichev, Degeorge-Patel-Zeckhauser)
and this engine detects it decisively (z +17.7, placebo *p* 0.0005). The *return penalty* is
directionally present but, on honest decomposition, roughly *half* of the naive penny-minus-decisive
spread is mechanical PEAD composition rather than a management penalty — and there is no free real
tape to test it on. `WEAK` × `MIRAGE`, discontinuity `CONFIRMED`. The synthetic control proves the
engine would catch a genuine penalty; the ceiling is set by the missing data and the composition
confound, not by the code.
