# Results — Study 552 (App-Store-Rankings): app-ranking momentum as a cross-sectional nowcast

*Generated from [`app_store_rankings/`](../app_store_rankings/) on the **deterministic synthetic
panel** (seed = 552). **No real tape was tested** — a free, point-in-time, survivorship-clean App
Store ranking panel joined to public tickers does not exist for a no-key retail stack (see the
data-availability wall below), so this study is **synthetic-only** and is capped at `WEAK`.
As-of **2026-06-30**.*

## The verdict, earned — Signal `WEAK` · Tradability `MIRAGE`

The claim: when a public consumer-tech company's app **climbs the App Store charts**, its stock
climbs with it — app-download/ranking momentum as a real-time fundamental *nowcast*, the broad
cross-sectional cousin of the single-name [294 Coinbase-Rank](../../294-coinbase-rank/) omen. We
build a cross-sectional **information-coefficient** engine (Spearman rank correlation of a
ranking-improvement signal with forward returns), a long-short tercile spread reported gross AND
net of costs + short borrow, a label-shuffle placebo null, a multi-window robustness sweep, and a
seed-robust synthetic positive control.

Because **no real tape exists**, the honest headline is the engine run on the **null synthetic
world** (the effect knob off) — what an alt-data desk would measure if the ranking signal carried
no information. It is, correctly, **flat**: pooled mean IC **+0.010** (*t* **+0.39**), long-short
**−0.5%/mo** gross (*t* −0.06), placebo *p* **0.70**, and the block-by-block IC **flips sign**
across the four windows. The positive control proves the *engine* is faithful: plant the effect
and the seed-averaged IC-*t* ramps smoothly (0.05→0.94, 0.10→1.76, **0.15→2.62**, 0.20→3.44,
0.30→5.13) and stays flat at the null (**+0.12**). So `WEAK` on the signal axis — the mechanism is
literature-plausible and the detector works, but there is **no real-tape confirmation and no
robust *t* ≥ 2 on any real data** — and `MIRAGE` on tradability (a monthly-rebalanced long-short
whose short leg is the sinking-app, small/hard-to-borrow tail; even the *planted*-effect spread
barely survives a punitive borrow, and the null-world spread is negative before costs).

## Data-availability wall — why this is synthetic-only

There is no free, survivorship-clean, point-in-time App Store ranking panel a retail stack can
reach:

- **Ephemeral.** Apple publishes only a live "top charts" snapshot; there is no official historical
  rank API.
- **Vendor-gated.** The usable history (App Annie / data.ai, Sensor Tower, Apptopia) is expensive,
  licensed, and itself *modelled/estimated* — not a clean tape.
- **Mapping-noisy.** An *app* is not a *ticker*: many chart-toppers are private, and a public
  parent's revenue is only partly the app (Uber ride-hail vs Eats, Meta's app family, …).

``fetch_panel(fetch=True)`` therefore **raises** rather than fabricate a "real" tape. This limitation
is stated on the SIGNAL axis; a synthetic-only study can never earn `REAL`.

## Data stamp (synthetic, deterministic)

| Object | Shape | Fingerprint |
|---|---|---|
| Null history (rank_alpha 0.0, 48 months × 30 names) | 1440 rows | `fc780c9f143d` |
| Planted history (rank_alpha 0.15, 48 months × 30 names) | 1440 rows | `5b92f4d9fdb5` |
| Single null cross-section | 30 rows | `ba6bda03b31d` |
| Single planted cross-section | 30 rows | `0d62f8a8066a` |

## The honest headline — the null synthetic world (no real tape)

| Metric | Value | Reads as |
|---|--:|---|
| Pooled mean IC (Spearman) | **+0.010** | ≈ 0 |
| IC *t*-stat (mean / SE over 48 monthly ICs) | **+0.39** | far below the *t* ≥ 2 bar |
| IC hit-rate (months with IC > 0) | **54.2%** | ~ coin flip |
| Long-short spread (top − bottom tercile), gross | **−0.53%/mo** | wrong sign, tiny |
| Long-short *t* | **−0.06** | noise |
| Label-shuffle placebo *p* | **0.70** | squarely in the null |

The signal carries no information in the null world, exactly as it should. This is the stand-in for
"what the real tape would have to beat" — and there is no real tape to beat it with.

## Robustness — the null sign is not stable (as expected for noise)

| Block | Mean IC | IC *t* |
|---|--:|--:|
| months 0-11 | +0.011 | +0.20 |
| months 12-23 | **−0.065** | −1.13 |
| months 24-35 | +0.067 | +1.46 |
| months 36-47 | +0.028 | +0.56 |

A signal whose block-by-block sign wanders around zero is noise — the correct null behaviour.

## Costs + borrow (monthly long-short)

| | value |
|---|--:|
| Gross spread (null world) | **−0.53%/mo** |
| Net (10 bps/leg round-trip × 4 crossings + 100 bps/yr borrow on the short leg) | **−1.02%/mo** |

Monthly rebalance charges the long AND short legs on entry and exit each month, plus a punitive
borrow on the sinking-app short leg (exactly the small/illiquid/hard-to-borrow tail). Even in the
*planted* world the net spread (**+21.2%/mo** gross **+21.7%/mo**) is dominated by the effect size;
in the null world the trade simply bleeds. `MIRAGE`.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `rank_alpha` | Mean IC-*t* (25 seeds) | |
|---|--:|---|
| 0.00 (null) | **+0.12** | flat — no false signal |
| 0.05 | +0.94 | too weak to see |
| 0.10 | +1.76 | emerging |
| **0.15** | **+2.62** | clears the *t* ≥ 2 bar |
| 0.20 | +3.44 | clear |
| 0.30 | +5.13 | strong |

At the null the IC-*t* is ≈ 0; planting a genuine ranking→return effect drives it smoothly past the
inference bar as the effect grows. The detector works — so the flat null result is a property of the
*data* (no signal planted), not a broken engine.

### Signal-noise sensitivity — blurrier alt-data, weaker signal (planted `rank_alpha` = 0.15)

| `signal_noise` (how noisy the rank read is) | Mean IC-*t* (25 seeds) |
|---|--:|
| 0.5 (clean read) | +3.33 |
| 1.0 (baseline) | +2.62 |
| 2.0 (noisy) | +1.70 |
| 4.0 (very noisy) | +0.92 |

The real-world catch: App Store rank is a *noisy* read on demand (discrete, capped at #1, only
partly the public parent's revenue). As the read gets blurrier the same true effect **falls below
the bar** — a concrete reason a genuine mechanism can still fail to certify on real data.

## The honest takeaway

The mechanism — app-ranking momentum nowcasting consumer-tech demand and, if the market is slow,
forward returns — is plausible and has serious literature behind alternative data (see
[references.md](references.md)). Our engine **proves it is detectable** when planted, and stays flat
when it is not. But: (1) **no free real tape exists** to test it on, so there is no robust *t* ≥ 2
on real data — `WEAK`, not `REAL`, by rule; (2) even the *planted* signal degrades below the bar as
the rank read gets noisy, which is the real-world regime; and (3) the long-short is un-tradable once
you pay a monthly rebalance and a borrow on the hard-to-short sinking-app leg — `MIRAGE`. Signal
`WEAK` × Tradability `MIRAGE`.
