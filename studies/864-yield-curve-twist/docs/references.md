# References & literature map — Study 864 (Yield-Curve Twist / Butterfly)

## The claim under test

- **Level, slope, curvature — the three modes of the curve.** **Litterman, R. & Scheinkman, J.
  (1991)**, *"Common Factors Affecting Bond Returns"* (Journal of Fixed Income). A principal-
  components decomposition of the Treasury curve finds three dominant factors that explain nearly
  all of its variation: **level** (a parallel shift), **slope** (a steepening/flattening), and
  **curvature** (a *butterfly* — the belly moving relative to the two wings). This study isolates
  that third factor and asks whether it *predicts* returns, not just explains variance.
- **The butterfly / twist trade.** A standard rates-desk construction is the *2-5-10* or *5-10-30*
  **butterfly**, `fly = 2·y_belly − y_short-wing − y_long-wing`; a positive fly means the belly
  yield sits *above* a straight line between the wings (the belly is *cheap*). A "twist" of the
  curve is a **change** in that curvature. The folklore: a cheap belly mean-reverts, so a high fly
  should precede belly (10-year) bonds outperforming. We test the daily 5-10-30 fly built from the
  CBOE yield indices `^FVX`/`^TNX`/`^TYX`.
- **The specific test here.** Sort/regress forward **IEF** (7-10y), **TLT** (20+y) and **SPY**
  returns on the *lagged, standardised* butterfly level and its change, with a Newey-West *t*, an
  **incremental** regression that partials out the 5s10s slope and the level (the distinctness /
  dedup test), a three-era robustness cut, a permutation placebo, a costed timing overlay, and a
  seeded synthetic positive control.

## What we measure, and the honesty rails

- **The butterfly, no free model.** `fly = 2·y10 − y5 − y30` in yield %-points, rebuilt identically
  on the real and synthetic tapes; the *twist* is its first difference `dfly`. Both are rolling-252d
  z-scored so the loading reads in bps of forward return per +1σ of curvature.
- **Point-in-time, one documented lag.** The butterfly known at the close of `t−1` (`.shift(1)`)
  forms the signal; the forward return runs from the close of `t`. Zero look-ahead; the rolling
  z-score and rank are causal.
- **Robust inference, honestly discounted.** Newey-West (HAC, Bartlett, lag = horizon) *t* on the
  regression loading and the Q5−Q1 spread — overlapping forward returns are strongly serially
  correlated, so a plain *t* overstates significance. The synthetic **null** documents that even
  the HAC *t* over-rejects under a persistent regressor (null sd ≈ 1.3), so the real full-sample
  *t* is read as ~1.8 effective σ, not 2.3.
- **The dedup is a regression, not a claim.** The incremental joint fit (`fly` + `slope` + `level`)
  shows the curvature is *not* the 2s10s slope repackaged — but also that its own *t* falls to 1.98
  once level/slope are held fixed.
- **The timer is graded separately.** Costs are one-way × NAV per regime switch — the honest test of
  whether a marginal curvature signal survives friction.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent covariance
  (the HAC *t* used on the regression loading and the spread series).
- **Litterman, R. & Scheinkman, J. (1991)** — the level/slope/curvature factor decomposition that
  motivates treating the butterfly as a distinct third mode.
- **Wilson, E. B. (1927)** — score interval for a binomial share (a shared desk primitive).

## Data sources

- **yfinance daily closes** (`auto_adjust=True`, total-return): `^FVX` (5y), `^TNX` (10y),
  `^TYX` (30y) CBOE yield indices, plus `IEF`, `TLT`, `SPY`; 2002-07-31 → 2026-06-30, cached under
  this study's own `_cache/`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [66-inverted-yield-curve](../../66-inverted-yield-curve/) — the **inversion** of the curve (a
  recession bellwether), a *level/slope* story. This study sorts on **curvature**, the third factor,
  orthogonal to whether the curve is inverted.
- [132-yield-curve-steepener](../../132-yield-curve-steepener/) — the **2s10s slope** (steep vs
  inverted) timing TLT. Here the slope is a *control* that the butterfly is tested *against*: the
  incremental regression shows the slope is insignificant while the curvature retains its loading.
- [380-treasury-roll-down](../../380-treasury-roll-down/) — **roll-down / carry** (the return from a
  bond rolling down a static curve), a *level-of-yield* mechanism. The butterfly is a *shape*
  signal, not a carry signal — no roll assumption enters.
- [581-term-premium](../../581-term-premium/) — the **term premium** (the long yield minus expected
  short rates), the compensation for *duration* risk along the *level* dimension. Curvature is the
  belly-vs-wings *shape*, not the level of duration compensation.

None of the siblings sort on the **curvature / butterfly (`2·y10 − y5 − y30`)** — the third yield-
curve factor — which is this study's own axis.
