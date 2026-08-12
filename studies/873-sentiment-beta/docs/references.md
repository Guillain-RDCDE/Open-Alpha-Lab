# References & literature map — Study 873 (Sentiment Beta)

## The claim under test

- **The source papers.** Malcolm **Baker & Jeffrey Wurgler**, *"Investor Sentiment and
  the Cross-Section of Stock Returns"* (Journal of Finance, 2006) and *"Investor
  Sentiment in the Stock Market"* (Journal of Economic Perspectives, 2007). They build a
  composite **investor-sentiment index** (from the closed-end-fund discount, NYSE
  turnover, the number and first-day returns of IPOs, the equity share in new issues, and
  the dividend premium) and show that when it is high, the subsequent returns of
  **speculative, hard-to-value, hard-to-arbitrage** stocks are low — and vice versa.
- **Sentiment beta — the cross-sectional loading.** A stock's **sentiment beta** is how
  strongly its returns *co-move with* the sentiment index. Names with a **high sentiment
  beta** are exactly the speculative ones that ride euphoria up and give it back
  afterward: high sentiment beta should predict **lower** subsequent returns, so a long
  **low-beta** / short **high-beta** portfolio should earn a *positive* spread, and the
  gap should widen **after sentiment peaks**.
- **The behavioural reading.** In euphoric regimes, sentiment-sensitive investors bid up
  the names that most visibly co-move with the mood; limits to arbitrage keep them
  over-priced, so their forward returns disappoint. It is the same over-pricing-of-the-
  speculative-tail mechanism behind the MAX, lottery, and realized-skewness effects,
  measured here through the **time-series co-movement** with a sentiment gauge.
- **The specific test here.** We take the self-contained daily version. We proxy market
  sentiment with a **tradable high-minus-low-volatility spread** built from the panel
  (the daily return of the most-volatile / speculative tercile minus the least-volatile /
  safe tercile — one of the two proxies the brief names, the other being the inverse of
  VIX). We estimate each name's **252-day sentiment beta** to that gauge, sort the
  cross-section, and measure the forward return of the equal-weight long-low-beta /
  short-high-beta book — with a Newey-West *t*, a post-peak conditional cut, a permutation
  placebo, a two-era robustness cut, a costed timer, and a seeded synthetic positive
  control.

## What we measure, and the honesty rails

- **A tradable sentiment gauge, no external feed.** Rather than an off-line-unavailable
  composite index, sentiment is proxied by the panel's own **high-minus-low realized-vol
  spread** — a real, tradable daily series that rises in risk-on euphoria. Real data, not
  a fabrication; its low autocorrelation (−0.01) matches a genuine daily long-short.
- **Sentiment beta, vectorised.** For each name, the rolling 252-day OLS slope of its
  daily return on the gauge, computed via the covariance identity `cov(r,g)/var(g)` over
  rolling means (no per-date regression loop).
- **Point-in-time sort, one documented lag.** The ranking beta is **known at the close of
  `t-1`** (`.shift(1)`); the book is held on day `t`. Zero look-ahead. The gauge's own
  trailing-vol ranking is likewise lagged one day.
- **Robust inference.** Newey-West (HAC, Bartlett, 10-lag) *t* on the daily long-short
  spread — an overlapping-formation signal is serially correlated, so a plain *t* would
  overstate significance. A one-sample *t* and a pooled Welch *t* (low-beta book vs
  high-beta book) cross-check. A **1,000-permutation placebo** breaks the
  signal → forward-return link to confirm the spread isn't a lucky alignment of the sort.
- **Survivorship is named on the Signal axis.** The universe is a **current-membership**
  set of ~50 liquid mega-caps, pulled through the `quantlab.universe` survivorship guard
  (`allow_survivorship_bias=True`, an explicit opt-in). Delisted / de-rated names are
  absent, so the cross-sectional magnitudes are an **upper bound** — and the speculative-
  small-stock effect is least likely to appear on mega-caps.
- **The timer is graded separately.** Costs are one-way × NAV on the long-short book, and
  the short book pays borrow — the honest test of whether a small daily spread survives
  friction and the speculative leg's volatility.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the spread series).
- **Wilson, E. B. (1927)** — score interval for a binomial share.
- **Fama, E. & MacBeth, J. (1973)** — the cross-sectional-regression tradition behind
  sorting names on an estimated loading and testing the forward-return spread.

## Data sources

- **yfinance daily OHLCV** (`auto_adjust=True`, total-return), 50 liquid US large-caps,
  2010-01-04 → 2026-06-30, cached under `_cache/` through `quantlab.universe`.
- The sentiment gauge is derived from that same panel (high-minus-low-vol spread) — no
  additional feed. Jeffrey Wurgler publishes the original monthly BW index on his NYU
  Stern page (people.stern.nyu.edu/jwurgler), which cannot be fetched offline; the
  tradable proxy stands in.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [258-baker-wurgler](../../258-baker-wurgler/) — the **time-series / aggregate**
  Baker-Wurgler claim: a high sentiment *level* predicts a low *market* return next month.
  It sorts nothing in the cross-section. This study tests the **cross-sectional
  sentiment-beta** leg — which *names* under-earn, ranked by their co-movement with the
  gauge.
- [255-fear-greed-index](../../255-fear-greed-index/) — a market-timing signal off a
  composite fear/greed gauge, again a time-series call on the index, not a cross-sectional
  beta sort.
- [501-idiosyncratic-volatility](../../501-idiosyncratic-volatility/) — sorts on a name's
  **own residual volatility level**. Sentiment beta is the **co-movement** of a name with
  a market-wide sentiment *time series*; a high-idio-vol name uncorrelated with the
  speculative leg has a *low* sentiment beta — different axis.
- [330-low-volatility-anomaly](../../330-low-volatility-anomaly/) — sorts on a name's
  **total volatility level** (the low-vol / betting-against-beta effect). This study sorts
  on the **loading on a sentiment gauge**, not on the volatility level itself.

None of the siblings sort on a name's **time-series beta to a sentiment gauge** — the
Baker-Wurgler sentiment-beta signal — which is this study's own axis.
