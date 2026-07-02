# Results — Study 576 (Muni-Treasury-Ratio): the M/T ratio as a rich/cheap timer

*Generated from [`muni_treasury_ratio/`](../muni_treasury_ratio/) over this study's cached yfinance
tape: daily total-return prices for **MUB** (iShares National Muni Bond ETF) and **IEF** (iShares
7-10Y Treasury ETF) plus each ETF's trailing-12-month distribution yield, 2008-01-03 → 2026-06-29
(fingerprint `738df551786b`). The signal is the muni/Treasury distribution-yield ratio, z-scored
over a trailing 252-day window; the target is the forward muni-minus-Treasury excess return. The
analysis window drops the first year (trailing-yield warm-up), leaving **2009-06-30 → 2026-06-29,
4,275 trading days** (fingerprint `c3e43ab615e6`; 4,087 signal/forward-aligned observations at the
63-day horizon). As-of **2026-06-30**.*

## The verdict, earned — Signal `WEAK` · Tradability `MIRAGE`

Muni-desk folklore: the **muni-Treasury yield ratio** (the "M/T ratio") is a rich/cheap gauge that
**times** muni or duration exposure — when munis yield an unusually large fraction of Treasuries
(a *high* ratio) munis are *cheap* and should outperform; when the ratio is *low*, munis are *rich*
and should lag. We z-score the MUB/IEF distribution-yield ratio and test whether it predicts the
forward muni-minus-Treasury excess return.

**The sign is right — and that is almost the whole story.** The predictive slope is **positive at
every horizon** (a high ratio does precede muni outperformance), and the quintile spread is
economically sizeable: the cheapest-muni quintile (Q5) earns **+0.51%** forward excess vs the
richest quintile's (Q1) **−0.09%**, a **+0.60%** spread over 63 days. But the *honest* statistic —
a Newey-West (HAC) *t* on the predictive slope, which accounts for the heavy autocorrelation of
overlapping forward windows — is **+0.43** (h = 63) and never exceeds **+1.28** (h = 252) across the
horizon sweep. The eye-catching two-sample *t* of **+4.12** on the quintile spread is an **overlap
illusion**: on **non-overlapping** 63-day windows (65 independent observations) the very same spread
(+0.74%) carries a Welch *t* of just **+0.51**. So `WEAK` on the signal axis (right sign,
literature-supported, but no robust *t* ≥ 2 on the real tape once overlap is handled), and `MIRAGE`
on tradability (the spread nets to +0.35% after costs but rests on a statistically insignificant,
overlap-inflated edge measured on a *distribution-yield proxy*, not the tradable MMD/AAA-GO curve a
desk quotes).

## Data stamp

- **Prices**: MUB + IEF, daily total return (auto-adjust), 2008-01-03 → 2026-06-29
- **Yields**: trailing-12-month distribution yield per ETF (Σ last 365d of distributions / price)
- **Full tape** (yields + returns + ratio): fingerprint `738df551786b`
- **Analysis window** (2009-06-30 → 2026-06-29, `mt_ratio`/`muni_ret`/`tsy_ret`): fingerprint
  `c3e43ab615e6`, 4,275 days

The distribution-yield ratio averages **1.30** over the window (min 0.81, max 2.42). This sits
*above* the ~0.6-0.9 the MMD/Treasury *yield* ratio famously prints, because a distribution yield
is not a par yield — the trailing-12m proxy is levels-biased. The signal is the **trailing z-score**
of the ratio, so the level bias washes out and only *relative* rich/cheap moves drive the test; but
the proxy gap is a real limitation, named on the SIGNAL axis.

## The headline predictive test — right sign, sub-threshold *t*

Predictive OLS of forward excess (muni − Treasury, 63 trading days) on the trailing ratio z-score,
Newey-West HAC standard error:

| | value |
|---|---|
| Slope (forward excess per 1σ ratio z) | **+0.080%** / σ |
| HAC *t* on the slope | **+0.43** |
| R² | 0.003 |
| n (aligned obs) | 4,087 |

The slope is the *right* sign (higher ratio → munis outperform), but the HAC *t* is nowhere near
the *t* ≥ 2 bar. The near-zero R² says the ratio explains essentially none of the forward-excess
variance.

## The quintile spread — economically real, statistically an overlap illusion

| Ratio quintile (by trailing z) | Forward 63-day muni − Treasury excess |
|---|---|
| **Q5** (munis cheapest, highest ratio) | **+0.51%** |
| **Q1** (munis richest, lowest ratio) | **−0.09%** |
| **Spread (Q5 − Q1)** | **+0.60%** (naive two-sample *t* **+4.12**, placebo *p* 0.0005) |

The naive *t* = +4.12 and placebo *p* = 0.0005 look decisive — until you remember every 63-day
forward window overlaps its 62 neighbours, so the 4,087 observations are nowhere near independent.
**Re-run the identical sort on non-overlapping 63-day windows** (65 independent observations) and the
same +0.74% spread carries a Welch *t* of just **+0.51**. The apparent significance was overlap,
not signal.

## Robustness — the sign holds, the *t* never clears the bar

| Forward horizon | Slope | HAC *t* | Q5 − Q1 spread |
|---|---|---|---|
| 21 days | +0.026% / σ | **+0.39** | +0.17% |
| 63 days (headline) | +0.080% / σ | **+0.43** | +0.60% |
| 126 days | +0.169% / σ | **+0.59** | +1.45% |
| 252 days | +0.554% / σ | **+1.28** | +2.64% |

The direction is *stable and correct* at every horizon — the folklore's sign is not in doubt. What
is missing is significance: even the strongest horizon (1-year) tops out at HAC *t* = +1.28. The
spread grows with horizon (as a slow, mean-reverting ratio should), but so does the standard error.

## Costs

| | value |
|---|---|
| Gross Q5 − Q1 spread (63-day) | **+0.60%** |
| Net (3 bps/leg round-trip + 50 bps/yr Treasury-leg borrow, 0.25y hold) | **+0.35%** |

Costs are almost a footnote here — the spread clears them numerically. But a +0.35% net edge that
rests on a HAC *t* of +0.43 and a non-overlap Welch *t* of +0.51 is not a tradable signal; it is a
directionally-correct wobble inside the noise.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `timing_beta` | Mean HAC slope-*t* (25 seeds) | |
|---|---|---|
| 0.00 (null) | **+0.15** | flat — no false signal |
| +0.03 | +0.80 | signal emerging |
| +0.06 | +1.45 | visible |
| +0.10 | **+2.25** | clears the bar |
| +0.15 | **+3.13** | strong |

At the null the HAC slope-*t* is ≈ 0; planting a genuine timing effect (`timing_beta > 0`) drives
the slope positive and past +2 as it grows. The detector works — so the real-tape *t* = +0.43 is a
statement about **this tape and this proxy**, not a broken engine. (Control only; never cited for
the real-tape stamp.)

## Why the ratio doesn't certify here

1. **Overlap inflates the naive test.** Overlapping 63-day forward windows make 4,087 observations
   behave like ~65 independent ones. The HAC *t* (+0.43) and the non-overlap Welch *t* (+0.51) agree;
   only the naive quintile *t* (+4.12) is fooled.
2. **Distribution-yield proxy, not the MMD curve.** yfinance exposes no free muni *yield index*, so
   the ratio is built from trailing-12m ETF distributions, not the AAA-GO/MMD par curve a muni desk
   times off. The level is biased (mean 1.30 vs the ~0.85 MMD ratio); the z-score fixes the level
   but not the fact that this is a coarser instrument than the real signal.
3. **A slow signal with a wide error band.** The ratio mean-reverts over quarters, so the honest
   information horizon is long — and over long horizons the effective sample is small and the
   standard error large. The sign is right at 1 year (spread +2.64%) but only reaches *t* = +1.28.

## The honest takeaway

The muni-Treasury ratio times muni-vs-Treasury returns in **the right direction** — cheap munis do
tend to outperform — and that directional consistency across horizons is why this earns `WEAK`, not
`NONE`. But the effect never clears a robust *t* ≥ 2 on the real tape: the HAC *t* is +0.43 at the
headline horizon, +1.28 at best, and the flashy quintile *t* of +4.12 is an overlap artifact that
collapses to +0.51 on independent windows. `WEAK` × `MIRAGE`, on a distribution-yield proxy rather
than the tradable curve. The synthetic control confirms the engine would bank a real timing effect
— so this is the tape (and the proxy) talking, not the code.
