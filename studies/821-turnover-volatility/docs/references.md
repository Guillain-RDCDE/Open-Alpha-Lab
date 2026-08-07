# References & literature map — Study 821 (Turnover Volatility)

## The claim under test

- **The source paper.** Tarun **Chordia**, Avanidhar **Subrahmanyam** & V. Ravi
  **Anshuman**, *"Trading activity and expected stock returns"* (Journal of Financial
  Economics, 2001). Beyond the well-known negative relation between the *level* of
  trading activity (turnover, dollar volume) and expected returns, they document that
  the **variability** of trading activity carries its own robust **negative**
  cross-sectional premium: sorting on the **coefficient of variation of turnover**
  (std/mean over a trailing window), the high-variability names go on to earn **lower**
  returns, even after controlling for size, book-to-market, and the turnover level.
- **The liquidity-risk reading.** Unpredictable liquidity is a cost: an investor who
  may need to trade dislikes a name whose ability to trade cheaply comes and goes. Such
  erratic-turnover names are bid up (a liquidity-risk discount) and subsequently
  under-earn — a variability effect distinct from the *level* of liquidity.
- **The specific test here.** We take the self-contained daily version: sort a liquid
  US cross-section on its **trailing 63-day coefficient of variation of daily share
  turnover** and measure the forward return of the equal-weight long-low-vol /
  short-high-vol book, with a Newey-West *t*, a permutation placebo, a two-era
  robustness cut, a dollar-volume variant, a costed timer, and a seeded synthetic
  positive control. (Without shares-outstanding we use raw daily Volume as the turnover
  proxy; a fixed shares count makes turnover a constant rescaling of Volume, and the CV
  std/mean is scale-invariant, so the two coincide.)

## What we measure, and the honesty rails

- **Coefficient of variation of turnover, no free model.** For each name, the rolling
  `window`-day std/mean of daily turnover, computed vectorised with pandas rolling
  moments. Scale-invariant, so raw share Volume and any fixed-shares turnover give the
  identical signal.
- **Point-in-time sort, one documented lag.** The ranking signal is the trailing CV
  **known at the close of `t-1`** (`.shift(1)`); the book is held on day `t`. Zero
  look-ahead.
- **Robust inference.** Newey-West (HAC, Bartlett, 10-lag) *t* on the daily long-short
  spread — an overlapping-window signal is serially correlated, so a plain *t* would
  overstate significance. A one-sample *t* and a pooled Welch *t* (low-vol book vs
  high-vol book) cross-check. A **1,000-permutation placebo** breaks the
  signal → forward-return link to confirm the spread isn't a lucky alignment of the sort.
- **Survivorship is named on the Signal axis.** The universe is a **current-membership**
  set of ~50 liquid mega-caps, pulled through the `quantlab.universe` survivorship guard
  (`allow_survivorship_bias=True`, an explicit opt-in). Delisted / de-rated names are
  absent, so the cross-sectional magnitudes are an **upper bound**. Critically, the CSA
  variability premium is documented as a small/illiquid-stock phenomenon — a
  50-mega-cap survivor panel is exactly where it is *least* likely to appear.
- **The timer is graded separately.** Costs are one-way × NAV on the long-short book,
  and the short book pays borrow — the honest test of whether a small daily spread
  survives friction.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the spread series).
- **Wilson, E. B. (1927)** — score interval for a binomial share.
- **Amihud, Y. (2002)** — the illiquidity measure (|return| per dollar traded) tested in
  the sibling study 140; a *level* of price impact, not the *variability* of turnover.
- **Datar, V., Naik, N. & Radcliffe, R. (1998)** — the turnover *level* anomaly
  (high-turnover names under-earn), the *level* cousin of this study's *variability*
  signal (tested in sibling 141).

## Data sources

- **yfinance daily OHLC + Volume** (`auto_adjust=True`, total-return prices), 50 liquid
  US large-caps, 2010-01-04 → 2026-06-30, cached under `_cache/` through
  `quantlab.universe`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [141-turnover-anomaly](../../141-turnover-anomaly/) — the **level** of turnover
  (Datar-Naik-Radcliffe: high-turnover names under-earn). This study sorts on the
  **variability** (coefficient of variation) of turnover, a different moment of the same
  raw series and a different paper.
- [140-amihud-illiquidity](../../140-amihud-illiquidity/) — Amihud's price-**impact**
  illiquidity (|return| per dollar of volume), a *level* of illiquidity. This study
  measures the *dispersion of trading activity itself*, not price impact.
- [512-high-volume-premium](../../512-high-volume-premium/) — the return response after
  an **abnormal single-day volume spike**, an event/attention proxy. This study uses the
  trailing *coefficient of variation* of daily turnover, a smooth dispersion statistic,
  not a one-day event.

None of the siblings sort on the **coefficient of variation of a name's own daily
turnover** — the Chordia-Subrahmanyam-Anshuman variability signal — which is this
study's own axis.
