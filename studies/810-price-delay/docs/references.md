# References & literature map — Study 810 (Price Delay)

## The claim under test

- **The source paper.** Kewei **Hou & Tobias J. Moskowitz**, *"Market Frictions, Price
  Delay, and the Cross-Section of Expected Returns"* (Review of Financial Studies, 2005).
  They define a stock's **price delay** as the fraction of its market-driven return
  variation that shows up only in **lagged** market moves: regress the stock's return on
  the contemporaneous market plus several lags, and compare the R² with and without the
  lagged terms (`D = 1 − R²_restricted / R²_unrestricted`). Sorting on delay, the most
  delayed decile earns a large premium — on the order of ~10% per year gross — over the
  least delayed, **not** explained by size, liquidity, or the standard factors.
- **The economic reading.** Price delay proxies for **investor recognition / information
  frictions**: small, illiquid, thinly-covered, neglected stocks incorporate market-wide
  news slowly, and investors demand a premium for holding names whose prices are slow to
  reflect information (a Merton-1987 investor-recognition / limits-to-arbitrage story). The
  effect concentrates in exactly the corner of the market — micro-cap, low-analyst-coverage
  — that a mega-cap panel excludes.
- **The specific test here.** We take the self-contained weekly version: for each name
  regress its **weekly** return on the contemporaneous market (the equal-weight
  cross-sectional mean of the panel) plus **4 weekly lags**, over a trailing **1-year**
  window; form `delay = 1 − R²_contemp-only / R²_with-lags`; sort a liquid US cross-section
  **long high-delay / short low-delay**; and grade the forward return with a Newey-West *t*,
  a permutation placebo, a two-era cut, a costed timer, and a seeded synthetic control.

## What we measure, and the honesty rails

- **Delay from an R² ratio, no free model.** The measure is a pure comparison of two
  nested weekly regressions per name (contemporaneous market vs contemporaneous + 4 lags),
  vectorised across the cross-section because the market design is shared by all names.
- **Point-in-time sort, one documented lag.** The ranking signal is the delay **known at
  the close of week `t-1`** (`.shift(1)`); the book is held over week `t`. Zero look-ahead.
- **Robust inference.** Newey-West (HAC, Bartlett, 6-lag) *t* on the weekly long-short
  spread — an overlapping-formation signal is serially correlated, so a plain *t* would
  overstate significance. A one-sample *t* and a pooled Welch *t* (high-delay book vs
  low-delay book) cross-check. A **1,000-permutation placebo** breaks the signal → forward
  return link to confirm the spread isn't a lucky alignment of the sort.
- **Survivorship is named on the Signal axis.** The universe is a **current-membership**
  set of ~50 liquid mega-caps, pulled through the `quantlab.universe` survivorship guard
  (`allow_survivorship_bias=True`, an explicit opt-in). Delisted / de-rated names are
  absent, so the cross-sectional magnitudes are an **upper bound** — and the low-coverage
  names where delay is supposed to pay are absent by construction.
- **The timer is graded separately.** Costs are one-way × NAV per week on the long-short
  book, and the short book pays borrow — the honest test of whether a small weekly spread
  survives friction.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the spread series).
- **Wilson, E. B. (1927)** — score interval for a binomial share.
- **Merton, R. C. (1987)** — capital-market equilibrium with **incomplete information**;
  the investor-recognition premium Hou-Moskowitz invoke to explain why slow-to-price names
  should earn more.
- **Lo, A. & MacKinlay, A. C. (1990)** — lead-lag cross-autocorrelations between big and
  small stocks; the microstructure kinematics underneath the delay measure.

## Data sources

- **yfinance daily OHLC** (`auto_adjust=True`, total-return), 50 liquid US large-caps,
  2010-01-04 → 2026-06-30, resampled to weekly (W-FRI), cached under `_cache/` through
  `quantlab.universe`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [140-amihud-illiquidity](../../140-amihud-illiquidity/) — the **Amihud** price-impact
  ratio (|return| per dollar of volume), a *liquidity level*. Delay is a *diffusion-speed*
  measure read from a lagged-market regression, not a volume-scaled impact statistic —
  correlated in the cross-section (both tag frictional names) but a different construct.
- [379-etf-lead-lag](../../379-etf-lead-lag/) — a **sector-ETF → member** lead-lag *timing*
  signal (trade the constituent off the basket's move). This study does not trade on the
  lagged market move itself; it uses the *magnitude* of a name's lagged-market loading as a
  cross-sectional **sort key** for a return premium.
- [512-high-volume-premium](../../512-high-volume-premium/) — the **high-volume-return**
  premium (attention / visibility from a volume shock). Delay is the opposite pole:
  *low-visibility, slow-to-price* names, measured from a return-regression R² ratio, not a
  volume event.

None of the siblings sort on the **lagged-market R² ratio** — the Hou-Moskowitz delay
measure — which is this study's own axis.
