# Results — Study 559 (Dark-Pool-Ratio): the off-exchange-volume signal, on a synthetic panel

*Generated from [`dark_pool_ratio/`](../dark_pool_ratio/). **There is no free real per-name daily
dark-pool-ratio tape** a no-key retail stack can reach (FINRA ATS volume is weekly and lagged, the
large off-ATS internalised flow is excluded, clean daily DPR feeds are paywalled — see
[`dark_pool_ratio/data.py`](../dark_pool_ratio/data.py)), so this study is **synthetic-only**: it
proves the machinery and states the tape gap on the SIGNAL axis. A synthetic-only study can never
earn `REAL` (that needs a robust t ≥ 2 on a real tape). Deterministic, offline, seeded on 559.
As-of **2026-06-30**.*

## The verdict, earned — Signal `NONE` · Tradability `MIRAGE`

The folklore: when a bigger slice of a stock's volume prints **off-exchange** (in dark pools / ATSs
/ internalisers), that is *informed accumulation* — patient institutions building a position
quietly — and the stock drifts up. We build a synthetic dark-pool-ratio (DPR) panel against forward
returns, measure the information coefficient (IC), sort into a quintile long-short, run a
label-shuffle placebo, and prove the engine with a seed-robust positive control.

`NONE` on the signal axis is earned two ways. **(1) No real tape.** The one number that could earn
`REAL` — a robust IC-t ≥ 2 on a *real* DPR panel — cannot be computed, because that panel is not
freely available. **(2) The synthetic null is flat, and the naive signal is a confound.** In the
null world (DPR carries no forward information), the raw quintile IC is ≈ 0 at the base seed (IC-t
**+0.44**, placebo *p* **0.65**). Averaged over 25 seeds the raw null IC-t is **+0.77** — a *mild
false positive* — but that lift is **entirely the size confound** (big, liquid names internalise
more *and* drift): control for size and the null DPR slope-t collapses to **−0.19**. So the
"signal" a naive DPR sort sees at the null is contamination, not information. `MIRAGE` on
tradability follows: a long-dark/short-lit book whose edge is a confound, on a panel we can't even
build from real data.

The **positive control** is clean: plant a genuine DPR effect (`dpr_alpha = 0.06`) and the IC-t
rises to **+2.32** (25 seeds), staying **+1.39** even after controlling for size — the engine banks
a real DPR signal when one exists and stays flat when it doesn't. That is the whole point of a
machinery-only study: the harness is faithful, but the tape to run it on doesn't exist for free.

## Data stamp

- **Synthetic panel** (deterministic, seed 559, n = 300 names): dark-pool ratio, size proxy,
  liquidity proxy, forward return.
  - Null world (`dpr_alpha = 0`), base seed: fingerprint `11f28bf40b8c`
  - Planted world (`dpr_alpha = 0.06`), base seed: fingerprint `fbdeeb6093c5`
- **Real DPR panel**: *not available* — `fetch_panel()` returns an empty frame by construction.

## The primary null-world cross-section (base seed 559, n = 300)

| | value | reads as |
|---|---|---|
| Spearman IC (DPR vs forward return) | **+0.026** | ≈ 0 |
| IC *t* (Fisher-z) | **+0.44** | not significant |
| Quintile spread (long dark − short lit) | **+1.28%** | tiny |
| Two-sample *t* on the spread | **+0.75** | not significant |
| Label-shuffle placebo *p* (2000 perms) | **0.65** | deep in the noise |
| OLS DPR slope-*t* (raw) | **+0.47** | flat |
| OLS DPR slope-*t* (controlling for size) | **−0.46** | flat |

At the null the IC sits on zero and the placebo confirms it — exactly what a faithful engine should
report when the venue-of-execution carries no forward information.

## The confound — why a naive DPR sort *looks* like a signal (25 seeds)

The dark-pool ratio is correlated with **size / liquidity** in reality (large, liquid names
internalise more off-exchange), and size is itself a return driver. A naive DPR sort mistakes that
for signal:

| At the null (`dpr_alpha = 0`), 25 seeds | mean IC-*t* | reads as |
|---|---|---|
| Raw DPR sort | **+0.77** | a *mild false positive* |
| DPR slope controlling for size | **−0.19** | flat — the lift was the confound |

The whole apparent edge at the null is the size confound. This is the single most important honesty
point of the study: on real data you would have to *net out size/liquidity* before believing a DPR
IC, and the free tape to do that doesn't exist.

## Robustness — the IC moves monotonically with the planted truth (25 seeds)

| Planted `dpr_alpha` | mean IC | mean IC-*t* | reads as |
|---|---|---|---|
| 0.00 (null) | +0.044 | **+0.77** | flat (confound only) |
| 0.03 | +0.089 | +1.54 | emerging |
| 0.06 | +0.134 | **+2.32** | clears the bar |
| 0.09 | +0.178 | +3.11 | strong |
| 0.12 | +0.221 | +3.88 | strong |

The IC and its *t* rise monotonically with the planted effect — the detector responds to truth, not
to seeds.

## Costs

| | value |
|---|---|
| Gross quintile spread (null world, base seed) | **+1.28%** |
| Net (5 bps/leg round-trip + 50 bps/yr borrow on the lit short, 1y hold) | **+0.58%** |

Costs are almost a footnote: at the null there is no edge to charge them against, and the ~0.7pp
wedge (4 crossings × 5 bps + 50 bps borrow) eats half of the noise-level gross.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `dpr_alpha` | mean IC-*t* (25 seeds) | |
|---|---|---|
| 0.00 (null) | **+0.77** | flat (confound-only false positive) |
| 0.06 | **+2.32** | clears the bar |
| 0.12 | **+3.88** | strong |

Planting a genuine DPR→return effect drives the IC-t past 2; at the null it stays near zero (and the
residual lift is the documented size confound, which the size-controlled test removes). The engine
works — so the `NONE` verdict is a statement about **data availability and confounding**, not a
broken detector. (Control only; never cited for a real-tape stamp — none exists.)

## Why the folklore doesn't certify here

1. **No free real tape.** FINRA ATS volume is weekly and released on a two-to-four-week lag; the
   large off-ATS internalised flow (wholesalers) is not in it; clean daily DPR feeds are paywalled.
   A point-in-time daily DPR panel joined to forward returns is simply not buildable offline — so
   the one test that could earn `REAL` cannot be run.
2. **The signal is confounded with size/liquidity.** Off-exchange share rises with size and
   liquidity, both of which drive returns independently. A raw DPR IC is contaminated (null raw IC-t
   +0.77 → −0.19 once size is netted out). Any real study must control for the confound *first*.
3. **Direction is ambiguous even in theory.** High off-exchange share can mean patient institutional
   accumulation (bullish) *or* heavy retail internalisation / distribution (neutral-to-bearish). The
   sign of the folklore is not pinned down by microstructure theory — another reason a real,
   sign-tested tape would be needed before believing it.

## The honest takeaway

The dark-pool-ratio "informed accumulation" story is untestable on a free retail stack — the daily
panel doesn't exist — and where it *can* be probed (the synthetic world), the naive version is a
size/liquidity confound rather than a venue signal. `NONE` × `MIRAGE`. The synthetic positive
control confirms the machinery would catch a real DPR effect at IC-t +2.32 if one existed and a tape
to measure it on were available; neither is true here.
