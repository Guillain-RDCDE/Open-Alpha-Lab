# References & literature map — Study 808 (Continuing Overreaction)

## The claim under test

- **The source paper.** Suk-Joon **Byun, Sonya S. Lim & Sang Hyun Yun**, *"Continuing
  Overreaction and Stock Return Predictability"* (Journal of Financial and Quantitative
  Analysis / working-paper series, 2016). They build a **weighted signed-momentum**
  measure — for each stock, a recency-weighted sum of the **signs** of its recent monthly
  returns, with weights increasing toward the more recent months (`w_j = (n − j)`,
  normalised). A high positive score marks a **persistent recent up-streak** ("continuing
  overreaction"): investors keep pushing a name that has moved consistently in one
  direction. The measure predicts the cross-section **positively** in the short run
  (continuation, later followed by reversal), and it subsumes / strengthens plain
  past-return momentum because it counts the *consistency* of the run rather than its
  cumulative magnitude.
- **The behavioural reading.** Barberis-Shleifer-Vishny-style *continuing overreaction*:
  a run of same-signed returns feeds extrapolative demand, so a consistent streak keeps
  going before it eventually corrects. Weighting recent months more heavily captures the
  freshest, most-extrapolated part of the streak.
- **The specific test here.** We take the self-contained monthly version: for each name,
  the normalised recency-weighted sum of the **signs of its trailing 12 monthly returns,
  skipping the most recent month** (the 1-month reversal buffer); sort a liquid US
  cross-section on it and measure the forward-month return of the equal-weight
  long-high-CO / short-low-CO book, with a Newey-West *t*, a permutation placebo, a
  two-era robustness cut, a costed timer, and a seeded synthetic positive control.

## What we measure, and the honesty rails

- **Signed momentum, only the signs.** For each name, `CO = Σ_p w_p · sign(r_month)` over
  the trailing 12 monthly returns skipping the most recent, `w_p ∝ (p+1)` (oldest→newest),
  normalised to sum 1 — so `CO ∈ [−1, +1]` and depends on the *directions* of the monthly
  steps, never their size.
- **Point-in-time sort, one documented lag.** The ranking score uses monthly returns
  **through month `i−2`**; the most-recent month `i−1` is **skipped** (the standard
  short-term-reversal buffer); the book is held on month `i`. Zero look-ahead.
- **Robust inference.** Newey-West (HAC, Bartlett, 6-lag) *t* on the monthly long-short
  spread — an overlapping-formation signal is serially correlated, so a plain *t* would
  overstate significance. A one-sample *t* and a pooled Welch *t* (high-CO book vs low-CO
  book) cross-check. A **1,000-permutation placebo** breaks the signal → forward-return
  link to confirm the spread isn't a lucky alignment of the sort.
- **Survivorship is named on the Signal axis.** The universe is a **current-membership**
  set of ~50 liquid mega-caps, pulled through the `quantlab.universe` survivorship guard
  (`allow_survivorship_bias=True`, an explicit opt-in). Delisted / de-rated names are
  absent, so the cross-sectional magnitudes are an **upper bound**.
- **The timer is graded separately.** Costs are one-way × NAV per monthly rebalance, and
  the short book pays borrow — the honest test of whether a monthly spread survives
  friction.

## Shared method citations

- **Jegadeesh, N. & Titman, S. (1993)** — the (12,1) cross-sectional momentum baseline
  that the CO measure refines by weighting recent monthly signs.
- **De Bondt, W. & Thaler, R. (1985)** — long-horizon **overreaction / reversal**, the
  eventual counterpart to short-run continuation.
- **Barberis, N., Shleifer, A. & Vishny, R. (1998)** — the behavioural
  under-/over-reaction model that motivates "continuing overreaction".
- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the spread series).
- **Wilson, E. B. (1927)** — score interval for a binomial share.

## Data sources

- **yfinance daily OHLC** (`auto_adjust=True`, total-return), 50 liquid US large-caps,
  2010-01-04 → 2026-06-30, resampled to month-end returns, cached under `_cache/` through
  `quantlab.universe`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [507-cross-sectional-momentum](../../507-cross-sectional-momentum/) — plain **(12,1)
  past-return** momentum, which sorts on the *cumulative magnitude* of the trailing
  return. This study sorts on the **weighted signs** of the monthly steps (consistency,
  not size) — the Byun-Lim-Yun refinement.
- [508-momentum-crashes](../../508-momentum-crashes/) — the *conditional crash risk* of
  the momentum factor (its left tail after bear markets), not a signed-consistency signal.
- [196-long-term-reversal](../../196-long-term-reversal/) — the 3-5-year **reversal**
  (De Bondt-Thaler), the opposite horizon and sign; CO is a short-run continuation.
- [510-frog-in-the-pan](../../510-frog-in-the-pan/) — information **discreteness** (how
  *smoothly* the past return arrived), a path-smoothness modifier of momentum, not a
  recency-weighted count of monthly signs.

None of the siblings sort on the **recency-weighted sum of the signs of a name's own
monthly returns** — the continuing-overreaction signal — which is this study's own axis.
