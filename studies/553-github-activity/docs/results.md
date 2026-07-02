# Results — Study 553 (GitHub-Activity): open-source velocity as an innovation nowcast

*Generated from [`github_activity/`](../github_activity/) on the **deterministic synthetic panel**
(seed 553): 30 listed-tech firms × 40 quarterly periods, with the believers' effect planted at a
**modest, realistic** cross-sectional information-coefficient `ic = 0.03`. Panel fingerprint
`33d0732cb08f`. As-of **2026-06-30**.*

> **Data-availability limitation (stated up front, on the SIGNAL axis).** There is **no free,
> point-in-time, survivorship-clean mapping from GitHub orgs to tickers**. Firms rename/fork/archive
> repos and move code private; the GitHub feeds are rate-limited *current* snapshots, so a retail
> stack cannot reconstruct the commit/star velocity a trader would have seen at each past date. This
> study is therefore **synthetic-only** — and a synthetic-only study can **never** earn `REAL`
> (that needs a robust *t* ≥ 2 on a real tape). The ceiling here is `WEAK`.

## The verdict, earned — Signal `WEAK` · Tradability `MIRAGE`

The claim: a tech firm's **public GitHub velocity** (commit / merged-PR / new-star run-rate,
z-scored across the field) is a live *innovation-intensity* nowcast — surges foreshadow forward
returns. We plant that effect at a realistic alt-data strength (`ic = 0.03`) and test whether the
engine certifies it as bankable.

On the headline synthetic tape the effect is **present and the right sign, but does not clear the
significance bar**: the mean per-period Spearman IC is **+0.045** (*t* = **1.64**, n = 40 quarters),
the label-shuffle placebo *p* = **0.122** (not significant), and the long-top/short-bottom quintile
book earns **+14.1%/yr gross** at *t* **1.51**. Across sub-samples and tail cuts the IC stays the
right sign but its significance **flickers** (IC-*t* 0.89 first half, 1.37 second half; long-short
*t* ranges 0.71 → 2.25 depending on the split). So `WEAK` on the signal axis — a genuine but
sub-*t*-2, sample-fragile relation, on a tape that *cannot be real* — and `MIRAGE` on tradability
(you cannot build the point-in-time GitHub→ticker signal a backtest would need, and the trade adds
a short-borrow leg on top).

## Data stamp

- **Synthetic panel**: 30 firms × 40 quarters (1200 rows), planted `ic = 0.03`, seed 553,
  fingerprint `33d0732cb08f`
- **Null panel** (`ic = 0`, seed 553): mean-IC *t* = **+0.71** (flat — no false signal)

## The headline nowcast test — cross-sectional IC

| | value |
|---|---|
| Mean per-period Spearman IC (velocity → forward return) | **+0.045** |
| IC *t*-stat (mean / (std / √40)) | **+1.64** (needs ≥ 2 for a bankable signal) |
| Fraction of quarters with IC > 0 | **60%** |
| Label-shuffle placebo *p* | **0.122** (not significant) |

The IC is the right sign (positive: higher velocity → higher forward return) and the plant is real,
but *t* = 1.64 is **below the 2 bar** and the placebo *p* = 0.122 says this sample is consistent with
noise. A realistic alt-data IC is small — and small ICs need long, clean tapes to certify. We do not
have one (and cannot get one — see the limitation).

## The decile/quintile long-short book

| | value |
|---|---|
| Long top-quintile / short bottom-quintile velocity, gross | **+14.1%/yr** (*t* 1.51) |
| Net (10 bps/leg per quarterly rebalance + 100 bps/yr short borrow) | **+12.3%/yr** |

The spread is positive but its *t* (1.51) does not clear 2 either; costs are a second-order drag
here, but the trade is not certified before you pay them.

## Robustness — the sign holds, the significance flickers

| Sample | Split | Mean IC | IC *t* | Long-short (ann) | LS *t* |
|---|---|---|---|---|---|
| full | deciles | +0.045 | +1.64 | +11.3% | +0.71 |
| full | quintiles | +0.045 | +1.64 | +14.1% | +1.51 |
| full | terciles | +0.045 | +1.64 | +15.1% | +2.25 |
| first half | quintiles | +0.030 | +0.89 | +5.5% | +0.45 |
| second half | quintiles | +0.059 | +1.37 | +22.7% | +1.59 |

The IC is **positive in every cut** (encouraging), but it only crosses *t* = 2 at the barely-sorted
tercile split, and each half of the sample on its own is insignificant. A signal whose significance
depends on the split and the sub-sample is `WEAK`, not `REAL` — and here it *couldn't* be `REAL`
regardless.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `ic` | Mean IC-*t* (25 seeds) | |
|---|---|---|
| 0.00 (null) | **+0.22** | flat — no false signal |
| 0.02 | +0.82 | emerging |
| 0.03 (headline) | +1.14 | present, sub-bar |
| 0.05 | +1.77 | approaching the bar |
| 0.10 | **+3.38** | clears the bar |

At the null the mean IC-*t* is ≈ 0; a genuinely strong planted nowcast (`ic ≥ 0.10`) drives the
mean IC-*t* past 2. The detector works — so the headline `WEAK` is a statement about a *realistic
alt-data IC on a finite sample*, not a broken engine. (Control only; never cited for a real-tape
stamp, because there is no real tape.)

## Why this can't be certified

1. **No point-in-time tape.** The GitHub→ticker velocity a trader would have seen at each past date
   cannot be reconstructed on a retail stack (renames, archives, private moves, rate limits,
   snapshot-only feeds). Any real backtest would be survivorship- and look-ahead-contaminated.
2. **A realistic alt-data IC is small.** Even *if* the nowcast is real, an IC of ~0.03–0.05 needs
   hundreds of clean cross-sections to clear *t* = 2 — a data budget the free feeds don't supply.
3. **Velocity ≠ value.** Public commit/star velocity conflates genuine shipping with open-source
   *marketing* (docs, demos, star-farming) and mix shifts (firms open-sourcing more of an existing
   product), so the observable proxy is noisier than the latent "innovation" the claim invokes.

## The honest takeaway

Engineering-as-alt-data is a *plausible* nowcast — the planted effect is the right sign and the
engine recovers it — but on a realistic-strength, finite tape it lands at IC-*t* **1.64** (placebo
*p* 0.12), sign-stable yet significance-fragile across cuts, and it can **never** be certified
`REAL` because the point-in-time GitHub→ticker tape it would need does not exist on a free stack.
`WEAK` × `MIRAGE`.
