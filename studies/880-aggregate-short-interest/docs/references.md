# References & literature map — Study 880 (Aggregate Short Interest)

## The claim under test

- **The source paper.** David E. **Rapach, Matthew C. Ringgenberg & Guofu Zhou**,
  *"Short Interest and Aggregate Stock Returns"* (Journal of Financial Economics,
  121(1), 2016, pp. 46–65). Aggregating short interest across all NYSE/AMEX/NASDAQ
  stocks and detrending it, they find the resulting **short-interest index (SII)** is
  "arguably the strongest known predictor of the equity risk premium" at the monthly
  horizon: a high, detrended aggregate short-interest reading forecasts **lower**
  forward market returns, with a strong **negative** in-sample slope, meaningful
  out-of-sample R², and economic value to a mean-variance timer. The mechanism they
  argue is that informed short sellers, in aggregate, anticipate declines in expected
  cash-flow growth.
- **The economic reading.** Short sellers are, on average, informed (Boehmer-Jones-Zhang);
  when they crowd the *whole* market rather than single names, that aggregate positioning
  is a market-level bearish signal, distinct from the noisy cross-sectional short-interest
  sort. Detrending is essential — raw aggregate short interest has a strong secular drift
  (rising with the growth of hedge-fund and ETF-arbitrage shorting) that must be removed
  before it predicts anything.
- **The specific test here.** We rebuild an aggregate index from the **FINRA Consolidated
  Short Interest** file — the official, public, bi-monthly settlement-date report — for a
  fixed liquid 50-name US panel, taking the equal-weight cross-sectional mean
  **days-to-cover** (short interest / average daily volume) as the market short-interest
  ratio, detrend its log against a linear trend (the RRZ step), and regress forward SPY
  total-return on it with a Newey-West slope *t*, a permutation placebo, a two-era cut, a
  costed timer, and a seeded synthetic positive control.

## What we measure, and the honesty rails

- **Frequency & lag.** Aggregate short interest is **bi-monthly** (FINRA settles the 15th
  and the last business day, ~24 prints/yr) and each print is **published ~8 business days
  after** its settlement date. So the sample is 205 observations since 2017-12, and a
  settlement-`t` reading is acted on only at the **next** settlement `t+1` — one documented
  execution lag, zero look-ahead.
- **Days-to-cover, not shares-outstanding.** FINRA publishes the short position and the
  average daily volume (hence days-to-cover) but **not** shares outstanding, so our index is
  a short-interest ratio in the *volume* sense rather than the *shares-outstanding* ratio of
  the original paper. Stated on the Signal axis.
- **Detrend caveat.** The linear trend is fit **in-sample** over the whole tape (a mild
  look-ahead confined to the trend estimate, standard in the predictive-regression
  literature); the two-era cut is the out-of-sample honesty check. Because a full-sample
  detrend can only *help* find a signal, a flat result under it is conservative.
- **Survivorship.** The 50-name panel is **current-membership** liquid mega-caps, so it omits
  delisted / de-rated names and understates the breadth of the market-wide index RRZ built.
  Named on the Signal axis.
- **Robust inference.** Newey-West (HAC, Bartlett, 6-lag) *t* on the slope — overlapping
  forward horizons are serially correlated, so a plain *t* would overstate significance. A
  5,000-draw permutation placebo breaks the signal → forward-return link to confirm the slope
  is not a lucky alignment.
- **The timer is graded separately.** A market-timing overlay (de-risk to cash on crowded
  shorts) is charged one-way × NAV per switch — the honest test of whether a weak monthly
  signal can beat buy-and-hold.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* on the regression slope).
- **Wilson, E. B. (1927)** — score interval for a binomial share.
- **Boehmer, E., Jones, C. & Zhang, X. (2008)** — short sellers are informed (the
  micro-foundation for treating aggregate short interest as an informed bearish signal).

## Data sources

- **FINRA Consolidated Short Interest**, public Query API
  (`https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest`), no key,
  per-name bi-monthly settlement records 2017-12-29 → 2026-06-30, cached under `_cache/`.
- **yfinance daily SPY** (`auto_adjust=True`, total-return), cached under `_cache/`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [262-short-interest](../../262-short-interest/) — the **cross-sectional** short-interest
  sort (do *individual* heavily-shorted names bounce or bleed?). This study is the
  **aggregate / time-series** cousin: one market-wide index predicting the *market*, the
  RRZ signal, not a stock-by-stock sort.
- [557-borrow-fee-signal](../../557-borrow-fee-signal/) — the **cost to borrow** (loan fee)
  as a short-demand signal on individual names, not the market-wide short-position ratio.
- [558-failures-to-deliver](../../558-failures-to-deliver/) — **settlement failures** (FTDs),
  a plumbing/short-pressure microstructure signal, not aggregate reported short interest.
- [260-margin-debt](../../260-margin-debt/) — aggregate **margin debt** (leverage on the
  *long* side) as a market timing gauge; this study is the mirror on the **short** side.

None of the siblings run the **market-wide short-interest index → aggregate market return**
predictive regression of Rapach-Ringgenberg-Zhou — this study's own axis.
