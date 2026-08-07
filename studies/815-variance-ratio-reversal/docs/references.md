# References & literature map — Study 815 (Variance-Ratio Reversal)

## The claim under test

- **The source paper.** Andrew W. **Lo & A. Craig MacKinlay**, *"Stock Market Prices Do
  Not Follow Random Walks: Evidence from a Simple Specification Test"* (Review of
  Financial Studies, 1988). They introduce the **variance ratio**
  `VR(q) = Var(q-day return) / (q × Var(1-day return))` and its overlapping,
  heteroskedasticity-robust estimator. Under the random-walk null `VR(q) = 1` for all
  `q`; `VR > 1` signals **positive** return autocorrelation (trending), `VR < 1` signals
  **negative** autocorrelation (mean reversion). Their weekly US index data rejects the
  random walk with `VR > 1` at short horizons.
- **The reversal reading.** If `VR < 1` names are genuinely mean-reverting, a
  cross-sectional book that goes **long the low-VR** names and **short the high-VR** names
  should harvest the reversal — the microstructure cousin of short-term reversal
  (Jegadeesh 1990; Lehmann 1990), but sorted on the *autocorrelation structure* itself
  rather than the level of the last return.
- **The specific test here.** We take the self-contained daily version: sort a liquid US
  cross-section on its **trailing 120-day `VR(q=5)`** and measure the forward return of the
  equal-weight long-low-VR / short-high-VR book, with a Newey-West *t*, a permutation
  placebo, two-era and two-window robustness cuts, a costed timer, and a seeded synthetic
  positive control. (Daily close-to-close sampling with `q=5` is a coarser random-walk
  probe than the paper's index-level weekly data, so the cross-sectional magnitudes here
  are conservative.)

## What we measure, and the honesty rails

- **Lo-MacKinlay variance ratio, no free model.** For each name, the rolling `window`-day
  overlapping VR of daily simple returns, bias-corrected by
  `m = q(W−q+1)(1−q/W)`, computed vectorised from rolling sums (unit-tested against the
  closed-form scalar estimator and against the MA(1) sign).
- **Point-in-time sort, one documented lag.** The ranking signal is the trailing VR
  **known at the close of `t−1`** (`.shift(1)`); the book is held on day `t`. Zero
  look-ahead.
- **Robust inference.** Newey-West (HAC, Bartlett, 10-lag) *t* on the daily long-short
  spread — an overlapping-window statistic is serially correlated, so a plain *t* would
  overstate significance. A one-sample *t* and a pooled Welch *t* (low-VR book vs high-VR
  book) cross-check. A **1,000-permutation placebo** breaks the signal → forward-return
  link to confirm the spread isn't a lucky alignment of the sort.
- **Survivorship is named on the Signal axis.** The universe is a **current-membership**
  set of ~50 liquid mega-caps, pulled through the `quantlab.universe` survivorship guard
  (`allow_survivorship_bias=True`, an explicit opt-in). Delisted / de-rated names are
  absent, so the cross-sectional magnitudes are an **upper bound**.
- **The timer is graded separately.** Costs are one-way × NAV on the long-short book, and
  the short book pays borrow — the honest test of whether a small daily spread survives
  friction.

## Shared method citations

- **Lo, A. W. & MacKinlay, A. C. (1989)** — *"The Size and Power of the Variance Ratio
  Test in Finite Samples: A Monte Carlo Investigation"* (Journal of Econometrics): the
  finite-sample behaviour of the estimator we use.
- **Jegadeesh, N. (1990)** & **Lehmann, B. (1990)** — short-term (one-month / one-week)
  return reversal, the economic effect the VR sort is a microstructure proxy for.
- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the spread series).
- **Wilson, E. B. (1927)** — score interval for a binomial share.

## Data sources

- **yfinance daily OHLC** (`auto_adjust=True`, total-return), 50 liquid US large-caps,
  2010-01-04 → 2026-06-30, cached under `_cache/` through `quantlab.universe`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [397-hurst-regime](../../397-hurst-regime/) — the **Hurst exponent** `H` from
  rescaled-range / detrended-fluctuation analysis, a *multi-scale* persistence estimate.
  The variance ratio is the Lo-MacKinlay **single-horizon** random-walk statistic (`q=5`),
  a different — and far more sampling-robust — memory diagnostic.
- [398-entropy-efficiency](../../398-entropy-efficiency/) — a **permutation-entropy**
  market-efficiency score built from ordinal patterns (an information-theoretic measure),
  not a second-moment variance ratio.
- [329-one-month-reversal](../../329-one-month-reversal/) — Jegadeesh (1990) sorts on the
  **level of the trailing one-month return** (the classic short-term reversal). This study
  sorts on the **shape of the return autocorrelation** — whether a name mean-reverts *at
  all* — not on the sign or size of its most recent move.

None of the siblings sort on the **Lo-MacKinlay overlapping variance ratio of a name's own
daily returns** — this study's own axis.
