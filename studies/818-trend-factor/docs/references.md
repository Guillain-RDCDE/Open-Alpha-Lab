# References & literature map — Study 818 (Trend Factor)

## The claim under test

- **The source paper.** Yufeng **Han**, Guofu **Zhou** & Yingzi **Zhu**, *"A Trend Factor: Any
  Economic Gains from Using Information over Investment Horizons?"* (Journal of Financial
  Economics, 2016; also RFS working versions). For each stock they form normalized
  moving-average signals `A_L = MA_L(price) / price` over short (3, 5, 10), intermediate
  (20, 50) and long (100, 200) day horizons, run a **cross-sectional (Fama-MacBeth) regression**
  of next-period return on the `A_L` vector each month, and average the estimated slopes over a
  trailing window. Dotting the averaged slopes into today's signals yields a fitted expected
  return — the **trend factor**. Sorting long-high / short-low on it, they report a large
  premium that **subsumes and outperforms** both single-horizon moving-average timing and the
  standard 12-1 momentum factor.
- **The economic reading.** Different investors react to price information over different
  horizons; a single moving average throws away all but one time scale. Letting a data-driven
  cross-sectional regression *weight* all the horizons at once is claimed to extract a cleaner,
  more persistent expected-return signal than any single-horizon rule — the "economic gains from
  using information over investment horizons" of the title.
- **The specific test here.** We take the self-contained daily version: build the seven
  `A_L = MA_L / price` signals, estimate the daily cross-sectional slopes, average the past
  slopes over a 250-day window, dot them into today's signals, and sort a liquid US
  cross-section long-high / short-low trend — with a Newey-West *t*, an explicit **contrast**
  against single-MA(200) timing and 12-1 momentum, a permutation placebo, a two-era robustness
  cut, a costed timer, and a seeded synthetic positive control. (Daily rebalancing on 50
  mega-caps is a thinner, more efficient slice than the paper's broad monthly CRSP panel, so a
  null here is a statement about *this* universe.)

## What we measure, and the honesty rails

- **Moving averages, no free model.** For each name and horizon the rolling `L`-day mean of the
  adjusted close, divided by today's price (`A_L = MA_L / P`) — a pure price transform.
- **Rolling Fama-MacBeth-lite slopes.** Each day, an OLS of the realized next-day return
  cross-section on the `A_L` vector (with intercept); the **expected** slope is the trailing
  250-day average of past slopes. The fitted trend factor is the dot product of that expected
  slope with today's `A_L` — exactly the paper's construction, at daily frequency.
- **Point-in-time sort, one documented lag.** The trend factor uses slopes and signals **known
  at the close of `t−1`** (`.shift(1)`); the book is held on day `t`. Zero look-ahead.
- **Robust inference.** Newey-West (HAC, Bartlett, 10-lag) *t* on the daily long-short spread —
  an overlapping, slowly-varying signal is serially correlated, so a plain *t* would overstate
  significance. A one-sample *t* and a pooled Welch *t* (high-trend book vs low-trend book)
  cross-check. A **1,000-permutation placebo** breaks the signal → forward-return link to confirm
  the (small, right-sign) spread isn't a lucky alignment of the sort.
- **Survivorship is named on the Signal axis.** The universe is a **current-membership** set of
  ~50 liquid mega-caps, pulled through the `quantlab.universe` survivorship guard
  (`allow_survivorship_bias=True`, an explicit opt-in). Delisted / de-rated names are absent, so
  the cross-sectional magnitudes are an **upper bound**.
- **The timer is graded separately.** Costs are one-way × NAV on the long-short book, and the
  short book pays borrow — the honest test of whether a small daily spread survives friction.

## Shared method citations

- **Fama, E. & MacBeth, J. (1973)** — the rolling cross-sectional-regression / averaged-slope
  procedure the trend factor's expected coefficients are built on.
- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the spread series).
- **Wilson, E. B. (1927)** — score interval for a binomial share.
- **Jegadeesh, N. & Titman, S. (1993)** — cross-sectional momentum, the 12-1 benchmark the
  trend factor is claimed to beat and which we run as a direct contrast.

## Data sources

- **yfinance daily OHLC** (`auto_adjust=True`, total-return), 50 liquid US large-caps,
  2010-01-04 → 2026-06-30, cached under `_cache/` through `quantlab.universe`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [110-faber-timing](../../110-faber-timing/) — a **single** 10-month moving-average *timing*
  rule (in/out of the market on one horizon). The trend factor is a **cross-sectional** sort on
  a data-weighted **blend of seven** horizons; we run the single-MA sort here only as a contrast,
  and it *beats* the blend on this universe.
- [438-triple-ma-crossover](../../438-triple-ma-crossover/) — a fixed **three-MA crossover**
  timing rule with hand-picked lengths and no cross-sectional regression. The trend factor lets
  a rolling Fama-MacBeth regression *weight* the horizons instead of hard-coding a crossover.
- [518-tsmom](../../518-tsmom/) — **time-series** (absolute) momentum: each asset long/short on
  its *own* past return sign. This study is **cross-sectional** and sorts on a fitted expected
  return from moving-average signals, not the own-return sign.
- [507-momentum](../../507-momentum/) — the classic **12-1 cross-sectional momentum** sort. We
  run exactly that as the second contrast; the trend factor is claimed to subsume it but here is
  *weaker* than it.

None of the siblings build the **rolling-Fama-MacBeth-weighted blend of seven normalized
moving-average horizons** — the Han-Zhou-Zhu trend factor — which is this study's own axis.
