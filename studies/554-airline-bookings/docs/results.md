# Results — Study 554 (Airline-Bookings): does booking momentum *lead* airline stocks?

*Generated from [`airline_bookings/`](../airline_bookings/) on the study's **synthetic** tape — a
deterministic monthly booking/return series (seed 554, 180 months, 2011-01 → 2025-12), frame
fingerprint `fb00515e49fb`. The headline world is the **efficient-market** case: bookings and the
same-month airline return share a demand shock (`contemp_beta = 0.9`) but bookings carry **no
forward information** (`lead_beta = 0`). No free, point-in-time flight-bookings index is
retail-reachable, so this study is **synthetic-only** and is capped at `WEAK` on the Signal axis
(a `REAL` stamp needs a robust t ≥ 2 on a real tape). As-of **2026-06-30**.*

## The verdict, earned — Signal `NONE` · Tradability `MIRAGE` · "Already in the price?" `CONFIRMED`

The alt-data pitch: a flight-**bookings momentum** signal turns up before airline earnings and
therefore before airline stock returns, so buying the sector when bookings are strong harvests a
predictable edge. We build the honest test — a predictive regression of *next*-month airline
return on *this*-month booking momentum, with an autocorrelation-robust (Newey-West) t, a
block-shuffle placebo, a timing rule net of costs and borrow, a sub-sample sweep, and a
seed-robust synthetic positive control.

On the headline (efficient-market) world the result is textbook **"already in the price."** The
**contemporaneous** regression is enormous — same-month airline return on booking momentum has
HAC *t* = **+12.6** (corr **+0.66**): plotted side by side, bookings look powerfully informative.
But the **predictive** regression — the only thing you can trade — is **noise**: slope HAC *t* =
**−0.07**, corr **−0.005**, placebo *p* = **0.94**. The lead is *contemporaneous*, not
*predictive*. So `NONE` on the signal axis (no forward edge; and synthetic-only caps it at `WEAK`
regardless), `MIRAGE` on tradability (the timing rule *loses* to buy-and-hold), and `CONFIRMED` on
the myth-check: the signal really is already embedded in the price.

## Data stamp

- **Synthetic booking/return series**: 180 monthly obs, 2011-01 → 2025-12, seed 554,
  `lead_beta = 0.0`, `contemp_beta = 0.9`, frame fingerprint `fb00515e49fb`
- **Real tape**: none — no free point-in-time flight-bookings index exists (`fetch_series`
  returns empty by default and raises on `fetch=True`); limitation named on the Signal axis

## The two regressions — the whole story

| Regression | slope | OLS *t* | HAC *t* | corr | reads as |
|---|---|---|---|---|---|
| **Contemporaneous** (same-month return on bookings) | +0.0553 | +11.66 | **+12.61** | **+0.66** | bookings *look* informative |
| **Predictive** (next-month return on bookings) | −0.0003 | −0.07 | **−0.07** | **−0.005** | nothing left to predict |

The gap between these two rows *is* the efficient-market trap. Bookings and prices genuinely move
together — the shiny same-month chart is real. But by the time a public reader sees the booking
print, the *forward* return carries no information: the market already discounted the demand.

## The placebo — the forward signal is indistinguishable from noise

Circular block-shuffle (block 6, 2000 permutations) of the booking signal against forward returns:
the observed predictive |HAC *t*| sits at the **94th** percentile of the *wrong* tail — placebo
*p* = **0.943**. A genuine lead would sit in the far tail (*p* < 0.05); this one is squarely in the
bulk. (Under a *planted* lead of `lead_beta = 0.75` the same placebo returns *p* = **0.0005** — so
the test has teeth; the headline world simply has no forward signal.)

## The timing rule — it loses to doing nothing

| Rule (held 1 month, one-way 10 bps) | Gross ann. | Net ann. | vs buy-and-hold (+3.3%/yr) |
|---|---|---|---|
| **Long/flat** (in iff bookings > 0) | −0.0% | **−0.5%** | **−3.8 pp/yr** (71 trades) |
| **Long/short** (short pays 300 bps/yr borrow) | −3.3% | **−5.6%** | **−8.9 pp/yr** (143 trades) |

Trading the booking signal *underperforms holding the basket* — before costs it is flat, after
costs and a short borrow it bleeds. There is nothing to harvest: `MIRAGE`.

## Robustness — the predictive *t* wanders around zero, both signs

| Window | slope | predictive HAC *t* | n |
|---|---|---|---|
| full | −0.000 | **−0.07** | 180 |
| first half | +0.004 | +0.70 | 90 |
| second half | −0.006 | −0.93 | 90 |
| third 1 | −0.002 | −0.27 | 60 |
| third 2 | +0.000 | +0.05 | 60 |
| third 3 | −0.004 | −0.51 | 60 |

No sub-sample clears |*t*| = 2; the sign flips between halves and thirds. A forward lead that is
not there in any window is not a signal.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `lead_beta` | Mean predictive HAC *t* (25 seeds) | |
|---|---|---|
| 0.00 (null) | **−0.35** | flat — no false lead |
| +0.25 | +2.75 | clears the bar |
| +0.50 | +5.77 | strong |
| +0.75 | +8.63 | very strong |
| +1.00 | +11.28 | saturated |

At the null the predictive HAC *t* is ≈ 0; planting a genuine forward lead (`lead_beta > 0`) drives
it cleanly past +2 and up. The detector works — so the headline "no forward edge" is a statement
about the *efficient-market world*, not a broken engine. (Control only; never cited for a real-tape
stamp — there is no real tape.)

## Why this can never certify `REAL`

1. **Synthetic-only, by data availability.** There is no free, point-in-time weekly/monthly
   flight-bookings index a retail reader can reach; the real vendors (card-panel aggregators,
   GDS/ARC ticketing feeds) are paywalled and revision-prone. A `REAL` stamp requires a robust
   *t* ≥ 2 on a *real* tape — impossible here, so the ceiling is `WEAK`. The honest run lands
   *below* that ceiling at `NONE` because the forward signal is noise.
2. **The efficient-market leakage is the whole point.** Even if a real panel *did* lead
   fundamentals, the market is forward-looking; the price can embed the booking signal before a
   public reader sees the print. This study shows exactly what that looks like: a huge
   *contemporaneous* correlation and a dead *forward* one.

## The honest takeaway

Rising flight bookings and airline stocks move together — that same-month correlation (HAC *t*
+12.6) is real and seductive. But it does not *lead*: on the forward return that you could actually
trade, booking momentum is noise (HAC *t* −0.07, placebo *p* 0.94), the timing rule loses to
buy-and-hold (−3.8 to −8.9 pp/yr), and the predictive *t* wanders around zero across every
sub-sample. `NONE` × `MIRAGE`, with the myth-check `CONFIRMED`: by the time you see the booking
signal, it is already in the price. The synthetic control proves the engine would catch a genuine
lead — so this is the (synthetic, efficient-market) tape talking, not the code.
