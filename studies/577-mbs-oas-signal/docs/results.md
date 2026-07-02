# Results — Study 577 (MBS-OAS-Signal): the mortgage-spread risk-off lead

*Generated from [`mbs_oas_signal/`](../mbs_oas_signal/) on this study's **deterministic synthetic
world** (there is no free agency-MBS OAS tape — see the data caveat below). Headline world: 779
weekly rows, 2011-01-07 → 2025-12-05, planted lead `lead_beta = -0.9`, panel fingerprint
`5a74734302c4`. Seed 577. As-of **2026-06-30**.*

## The verdict, earned — Signal `WEAK` · Tradability `MIRAGE`

The folklore: the agency-MBS **option-adjusted spread** (OAS) — the extra yield mortgage bonds pay
over Treasuries after removing the prepayment option — is a *leading* indicator of risk-off. When
OAS **widens**, mortgage investors are demanding more risk compensation, and that move is supposed
to front-run weakness in equities and corporate credit, so a rising OAS should predict *low* forward
returns.

We cannot test that on a real tape: **a clean, long, free OAS series does not exist** for a no-key
retail stack (the canonical series — ICE BofA US MBS OAS, Bloomberg — are licensed vendor products;
FRED exposes mortgage *yields*, not option-adjusted spreads). So this study is **synthetic-only**: a
deterministic generator plants (or withholds) an OAS→forward-return lead, and the engine measures
whether it can recover it.

On the planted world the engine **recovers the lead cleanly**: forward-return-on-OAS-change slope
**−0.88 %pt per +1 sd widening** (*t* **−11.4**, corr **−0.38**, R² **0.14**), label-shuffle placebo
*p* = **0.0005**; and it stays **flat at the null** (slope-*t* **+0.26**, placebo *p* **0.80** when
`lead_beta = 0`). A risk-off timing overlay lifts the Sharpe from **0.59** (buy-and-hold) to **1.07
net**. So the machinery is faithful and the folklore is *coherent* — but because there is **no real
OAS tape**, the study is capped at `WEAK` on the SIGNAL axis (a `REAL` stamp needs a robust *t* ≥ 2
on a genuine tape), and `MIRAGE` on tradability (the very series the signal is built from is
unreachable for a retail investor, so even a working lead cannot be harvested).

## Data stamp

- **Synthetic world**: 779 weekly (OAS, forward-return) rows, 2011-01-07 → 2025-12-05, seed 577,
  planted `lead_beta = -0.9`, fingerprint `5a74734302c4`
- **Real tape**: none — no free agency-MBS OAS series exists (named on the SIGNAL axis)

## The predictive relation — the sign IS the claim (planted world)

| | value |
|---|---|
| Slope (forward return on standardised weekly OAS change) | **−0.88 %pt** per +1 sd widening |
| Slope *t* | **−11.41** (a *negative* slope is the risk-off lead) |
| corr(OAS change, forward return) | **−0.38** |
| In-sample R² | **0.143** |
| Label-shuffle placebo *p* | **0.0005** |
| n (weeks) | **779** |

A widening in OAS this week is followed by a *lower* return next week — the folklore's negative lead,
recovered at *t* −11.4 on the world where it was planted.

## Null world — the engine does not hallucinate

| Planted `lead_beta` | Slope | Slope *t* | Placebo *p* |
|---|---|---|---|
| **0.0 (null)** | +0.020 | **+0.26** | **0.80** |
| −0.9 (headline) | −0.880 | −11.41 | 0.0005 |

With the lead switched off the slope-*t* collapses to ≈ 0 and the placebo *p* is a boring 0.80 — no
false signal.

## The timing overlay — a tradable risk-off switch (planted world)

Step to cash the week *after* OAS widens by more than 1 sd; otherwise hold the risk asset. Costs 5
bps one-way on every switch.

| | Buy & hold | Timed (gross) | Timed (net) |
|---|---|---|---|
| CAGR | **8.8%** | **16.7%** | **16.2%** |
| Sharpe | **0.59** | **1.10** | **1.07** |

68 weeks in cash, 131 switches. On the world where the lead is real, the switch works and survives
costs — but this is the *planted* world, not a tape you could trade.

## Robustness — the change leads, the level does not

| Signal definition | Slope | Slope *t* | Reads as |
|---|---|---|---|
| **Weekly change (headline)** | **−0.880** | **−11.41** | lead present |
| Level z-score | −0.024 | **−0.29** | no lead |
| 4-week change | −0.352 | **−4.25** | lead present (weaker) |

The *change* in OAS leads; the *level* does not — an honest nuance the generator faithfully carries
(the plant is on the standardised change). A widening move, not a wide level, is the signal.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `lead_beta` | Mean slope-*t* (25 seeds) | |
|---|---|---|
| 0.0 (null) | **+0.04** | flat — no false signal |
| −0.3 | −3.74 | lead emerging |
| −0.6 | −7.53 | lead visible |
| −0.9 (headline) | **−11.32** | strong |
| −1.5 | −18.90 | very strong |

At the null the slope-*t* is ≈ 0; planting a genuine lead (`lead_beta < 0`) drives the slope-*t*
monotonically negative, averaged over 25 seeds so no lucky seed can fake it. The detector works —
which is exactly why the honest verdict is about **data availability**, not a broken engine.

## Why this is capped at `WEAK` × `MIRAGE`

1. **No real OAS tape.** The option-adjusted spread is a licensed vendor index; FRED's mortgage
   series are yields, not OAS. A `REAL` stamp requires a robust *t* ≥ 2 on a genuine tape — which we
   cannot reach. Synthetic-only ⇒ capped at `WEAK`.
2. **The series itself is unreachable to trade on.** Even if the lead were real, a retail investor
   cannot subscribe to the OAS feed, cannot trade the MBS index cheaply, and receives OAS with a
   publication lag. The signal's raw material is behind the same wall as the data.
3. **Fragile signal shape.** Even in the synthetic world only the *change* leads — the *level* is
   flat (*t* −0.29). A real OAS series would carry regime shifts, prepayment-model artefacts and
   look-ahead in the OAS computation itself that this clean generator does not.

## The honest takeaway

The MBS-OAS risk-off lead is a *plausible, literature-adjacent* story, and the engine here proves it
*would* be detectable (slope-*t* −11.4, placebo *p* 0.0005) and *would* time well (Sharpe 0.59 →
1.07) **if** the effect were real and the data reachable. Neither condition is met on a free retail
stack: there is no OAS tape to test, and no OAS feed to trade. `WEAK` (coherent + machinery faithful,
but no real tape) × `MIRAGE` (the input series is itself unreachable).
