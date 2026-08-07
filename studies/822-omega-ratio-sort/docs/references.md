# References & literature map — Study 822 (Omega-Ratio Sort)

## The claim under test

- **The source paper.** Con **Keating & William F. Shadwick**, *"A Universal Performance
  Measure"* (The Journal of Performance Measurement, 2002). They introduce **Omega**,
  `Ω(L) = E[max(r − L, 0)] / E[max(L − r, 0)]` — the ratio of probability-weighted gains
  above a threshold `L` to probability-weighted losses below it. Because it integrates the
  *entire* return distribution (both tails, every moment), Omega is pitched as a "universal"
  replacement for the Sharpe ratio, which discards everything past the mean and variance.
- **The sort.** Rank a cross-section on each name's trailing Omega and go **long high-Omega /
  short low-Omega**. At the natural threshold `L = 0`, Omega is a **gain/loss ratio**: the
  average up-day return divided by the average down-day loss. The pitch under test: seeing
  skewness and fat tails, this distribution-aware sort should out-perform a plain
  trailing-Sharpe sort.
- **The specific test here.** We take the self-contained daily version: sort a liquid US
  cross-section on its **trailing-year Omega(0)** (a 252-day window ending ~1 month ago, 12-1
  style) and measure the forward return of the equal-weight long-high / short-low book,
  head-to-head against the identical Sharpe sort, with a Newey-West *t*, a permutation placebo,
  a two-era robustness cut, a costed timer, and a seeded synthetic positive control.

## The honesty rails, and why Omega ≈ Sharpe here

- **`Ω(0)` is tied to the mean.** Since `E[max(r,0)] − E[max(−r,0)] = E[r]`, the gain/loss
  ratio exceeds 1 exactly when the window mean is positive and rises as the loss tail shrinks.
  On daily equity returns it is a near-monotone re-labelling of **mean-over-dispersion** — the
  same object Sharpe measures — which is why the two sorts come out ~0.99 rank-correlated. The
  study's job is to show that the "extra moments" claim does not survive contact with the tape.
- **Point-in-time sort, one documented lag.** The ranking signal is the trailing Omega **known
  at the close of `t-1`** (`.shift(1)`, on top of the 21-day formation skip); the book is held
  on day `t`. Zero look-ahead.
- **Robust inference.** Newey-West (HAC, Bartlett, 10-lag) *t* on the daily long-short spread —
  an overlapping-formation signal is serially correlated, so a plain *t* would overstate
  significance. A one-sample *t* and a pooled Welch *t* (high book vs low book) cross-check. A
  **1,000-permutation placebo** breaks the signal → forward-return link.
- **Survivorship is named on the Signal axis.** The universe is a **current-membership** set of
  ~50 liquid mega-caps, pulled through the `quantlab.universe` survivorship guard
  (`allow_survivorship_bias=True`, an explicit opt-in). Delisted / de-rated names are absent,
  so the cross-sectional magnitudes are an **upper bound**.
- **The timer is graded separately.** Costs are one-way × NAV on the long-short book, and the
  short book pays borrow — the honest test of whether a small daily spread survives friction.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the spread series).
- **Wilson, E. B. (1927)** — score interval for a binomial share.
- **Bernardo, A. & Ledoit, O. (2000)** — the **gain-loss ratio** as an asset-pricing bound;
  `Ω(0)` is precisely a realized gain-loss ratio, the empirical cousin of their measure.
- **Jegadeesh, N. & Titman, S. (1993)** — 12-1 momentum; the formation-window-with-skip
  convention we borrow so the Omega signal is comparable to the Sharpe/momentum literature.

## Data sources

- **yfinance daily OHLC** (`auto_adjust=True`, total-return), 50 liquid US large-caps,
  2010-01-04 → 2026-06-30, cached under `_cache/` through `quantlab.universe`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [814-trailing-sharpe-anomaly](../../814-trailing-sharpe-anomaly/) — the **direct
  comparator**: the same cross-sectional sort on trailing **Sharpe** (mean/std, first two
  moments only). This study's whole reason to exist is the head-to-head — and the answer is
  that Omega is ~0.99 rank-identical to Sharpe and does **not** beat it.
- [330-low-volatility-anomaly](../../330-low-volatility-anomaly/) — the **low-vol** tilt (sort
  on trailing volatility alone). Omega's denominator is a downside-loss measure, so a
  high-Omega book could be a hidden low-vol book; we measure the rank overlap (Omega ~ (−vol) =
  +0.075 here) to show it is **not** the confound — Omega tracks Sharpe, not raw low-vol.

Neither sibling sorts on the **realized gain/loss ratio of a name's own daily returns** — the
Keating-Shadwick Omega signal — which is this study's own axis; and only this study runs the
Omega-vs-Sharpe head-to-head that is the point.
