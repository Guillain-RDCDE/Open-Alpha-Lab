# References & literature map — Study 809 (Signed Jump Variation)

## The claim under test

- **The realized-semivariance decomposition.** Ole **Barndorff-Nielsen, Silja Kinnebrock &
  Neil Shephard**, *"Measuring Downside Risk — Realised Semivariance"* (in *Volatility and Time
  Series Econometrics*, 2010). They split realized variance into an **upside** and a
  **downside** semivariance by the sign of the return — `RS+ = Σ r²·1(r>0)` and
  `RS- = Σ r²·1(r<0)` — and show the two halves carry different information about jumps and
  future volatility. The **signed jump variation** `RS+ − RS-` isolates the sign of the
  jump/asymmetry component.
- **The cross-sectional pricing.** Tim **Bollerslev, Sophia Zhengzi Li & Bingzhi Zhao**, *"Good
  Volatility, Bad Volatility, and the Cross Section of Stock Returns"* (Journal of Financial and
  Quantitative Analysis, 2020). Building realized semivariances from high-frequency returns, they
  find the **signed jump variation** is priced **negatively**: stocks with more **downside**
  ("bad") volatility earn a **premium**, while the **upside** ("good") volatility names under-earn.
  A long low-signed-jump / short high-signed-jump portfolio earns a positive spread. Related work
  (Feunou, Jahan-Parvar & Tédongap; Patton & Sheppard) documents the same downside-variance
  asymmetry in aggregate and cross-sectional risk premia.
- **The behavioural / risk reading.** Downside variance is genuinely feared (crash risk), so
  bearing it is compensated; upside variance is lottery-like and over-paid-for, so it is
  negatively priced — the same tail-preference channel behind the MAX and realized-skewness
  effects, but read off the **sign-split of variance** rather than a moment.
- **The specific test here.** We take the self-contained daily version: sort a liquid US
  cross-section on its **trailing signed jump variation of daily returns** `(RS+ − RS-)/RV` and
  measure the forward return of the equal-weight long-low-SJ / short-high-SJ book, with a
  Newey-West *t*, a permutation placebo, a two-era robustness cut, a costed timer, and a seeded
  synthetic positive control. (Daily returns are a coarser semivariance estimator than the paper's
  5-minute intraday sampling, so the magnitudes here are conservative.)

## What we measure, and the honesty rails

- **Signed jump variation, no free model.** For each name, the rolling `window`-day realized
  semivariances `RS+`/`RS-` (sums of squared up-/down-day returns) and the scaled signed jump
  `SJ = (RS+ − RS-)/RV`, in `[-1, +1]`. Vectorised as rolling sums of sign-masked `r²`.
- **Point-in-time sort, one documented lag.** The ranking signal is the trailing signed jump
  **known at the close of `t-1`** (`.shift(1)`); the book is held on day `t`. Zero look-ahead.
- **Robust inference.** Newey-West (HAC, Bartlett, 10-lag) *t* on the daily long-short spread — an
  overlapping-formation signal is serially correlated, so a plain *t* would overstate significance.
  A one-sample *t* and a pooled Welch *t* (low-SJ book vs high-SJ book) cross-check. A
  **1,000-permutation placebo** breaks the signal → forward-return link to confirm the spread is
  not a lucky alignment of the sort.
- **Survivorship is named on the Signal axis.** The universe is a **current-membership** set of
  ~50 liquid mega-caps, pulled through the `quantlab.universe` survivorship guard
  (`allow_survivorship_bias=True`, an explicit opt-in). Delisted / de-rated names are absent, so
  the cross-sectional magnitudes are an **upper bound**.
- **The timer is graded separately.** Costs are one-way × NAV on the long-short book, and the short
  book pays borrow — the honest test of whether a small daily spread survives friction.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent covariance
  (the HAC *t* used on the spread series).
- **Wilson, E. B. (1927)** — score interval for a binomial share.
- **Patton, A. & Sheppard, K. (2015)** — *"Good Volatility, Bad Volatility: Signed Jumps and the
  Persistence of Volatility"* (Review of Economics and Statistics) — the semivariance / signed-jump
  machinery in a time-series volatility-forecasting context.

## Data sources

- **yfinance daily OHLC** (`auto_adjust=True`, total-return), 50 liquid US large-caps,
  2010-01-04 → 2026-06-30, cached under `_cache/` through `quantlab.universe`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [803-realized-skewness-reversal](../../803-realized-skewness-reversal/) — the trailing
  **realized skewness** (the standardised third moment `m3/m2^1.5`), a single asymmetry number.
  This study sorts on the **sign-split of variance** (`RS+` vs `RS-`), a different estimator (a
  *variance* decomposition, not a *moment*) and a different paper (Bollerslev-Li-Zhao, not Amaya).
- [505-left-tail-momentum](../../505-left-tail-momentum/) — a **left-tail** (downside VaR / worst
  daily return) *momentum* signal, a single extreme order statistic carried forward, not the
  up-vs-down **variance ratio** of the whole trailing month.
- [130-variance-risk-premium](../../130-variance-risk-premium/) — the market-level
  **implied-minus-realized** variance gap (a time-series *option* signal), not a cross-sectional
  sort on a name's own **realized** up/down variance split.

None of the siblings sort on the **realized signed jump variation `(RS+ − RS-)/RV` of a name's own
daily returns** — the Bollerslev-Li-Zhao signal — which is this study's own axis.
