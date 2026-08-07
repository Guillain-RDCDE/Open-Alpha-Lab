# References & literature map — Study 807 (Salience-Theory Returns)

## The claim under test

- **The source paper.** Mathijs **Cosemans & Rik Frehen**, *"Salience Theory and Stock Prices:
  Empirical Evidence"* (Journal of Financial Economics, 2021). Applying the salience model of
  choice to the cross-section, they show that a stock whose recent returns were **salient on the
  upside** — its most attention-grabbing days were *up* relative to the market — is over-valued by
  salience-thinking investors and goes on to earn **lower** returns; the salience-theory value ST
  is a robust **negative** predictor, and a long low-ST / short high-ST portfolio earns a positive
  spread that survives standard controls.
- **The model.** Pedro **Bordalo, Nicola Gennaioli & Andrei Shleifer**, *"Salience Theory of
  Choice Under Risk"* (Quarterly Journal of Economics, 2012) — states that stand out by their
  **contrast** with a reference (here the market return) are over-weighted in the decision. The
  salience of a payoff is `σ(rᵢ, rₘ) = |rᵢ − rₘ| / (|rᵢ| + |rₘ| + θ)` (θ a small floor); states are
  ranked by σ and given decision weights that **decline in salience rank** (`δ^rank`, δ<1), so the
  most salient day is over-weighted.
- **The behavioural reading.** A salient *upside* recent tape draws salience-loving demand that bids
  the name up, lowering its subsequent return — a tail-over-pricing mechanism cousin to the MAX and
  lottery-demand effects, but signed by whether the salient days beat *the market* and weighting
  *every* day by its contrast, not just the single extreme.
- **The specific test here.** We take the self-contained daily version: for each name build the
  trailing-21-day salience-theory value ST (θ=0.1, δ=0.7), sort a liquid US cross-section, and
  measure the forward return of the equal-weight long-low-ST / short-high-ST book, with a
  Newey-West *t*, a permutation placebo, a two-era robustness cut, a costed timer, and a seeded
  synthetic positive control. (Daily returns and a 50-name equal-weight market are a coarser
  salience input than the paper's broad cross-section, so the mega-cap magnitudes are conservative.)

## What we measure, and the honesty rails

- **Salience-theory value, no free model.** For each name, the trailing `window`-day
  salience-weighted mean of market-excess returns, with the market the equal-weight
  cross-sectional mean and the decision weights `δ^rank` over the salience ranking — computed
  vectorised via `sliding_window_view` + double `argsort` (no per-date Python loop over the panel).
- **Point-in-time sort, one documented lag.** The ranking signal is the ST **known at the close of
  `t-1`** (`.shift(1)`); the book is held on day `t`. Zero look-ahead.
- **Robust inference.** Newey-West (HAC, Bartlett, 10-lag) *t* on the daily long-short spread — an
  overlapping-formation signal is serially correlated, so a plain *t* would overstate significance.
  A one-sample *t* and a pooled Welch *t* (low-ST book vs high-ST book) cross-check. A
  **1,000-permutation placebo** breaks the signal → forward-return link to confirm the spread isn't
  a lucky alignment of the sort.
- **Survivorship is named on the Signal axis.** The universe is a **current-membership** set of ~50
  liquid mega-caps, pulled through the `quantlab.universe` survivorship guard
  (`allow_survivorship_bias=True`, an explicit opt-in). Delisted / de-rated names are absent, so the
  cross-sectional magnitudes are an **upper bound**.
- **The timer is graded separately.** Costs are one-way × NAV on the long-short book, and the short
  book pays borrow — the honest test of whether a small daily spread survives friction.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent covariance
  (the HAC *t* used on the spread series).
- **Wilson, E. B. (1927)** — score interval for a binomial share.
- **Barberis, N. (2013)** — *"Thirty Years of Prospect Theory in Economics"* — survey placing
  salience theory among the family of reference-dependent, probability-distorting decision models
  (the prospect-theory cousin tested in study 806).

## Data sources

- **yfinance daily OHLC** (`auto_adjust=True`, total-return), 50 liquid US large-caps,
  2010-01-04 → 2026-06-30, cached under `_cache/` through `quantlab.universe`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [806-prospect-theory-value](../../806-prospect-theory-value/) — the Barberis-Mukherjee-Wang
  **prospect-theory** value: an S-shaped, loss-averse, *probability-weighted* valuation of a name's
  own return distribution. Salience theory weights days by their **contrast with the market
  return** (a relative-salience ranking), not by a fixed value/weighting function — a different
  behavioural primitive and a different paper.
- [365-lottery-max-effect](../../365-lottery-max-effect/) — the single **maximum daily return**
  (MAX) over a month, one extreme order statistic. ST weights *every* day by its salience and is
  signed by whether the salient days were up or down versus the market, not by the lone maximum.
- [503-expected-idiosyncratic-skewness](../../503-expected-idiosyncratic-skewness/) — a **modelled
  ex-ante** idiosyncratic-skewness forecast (Boyer-Mitton-Vorkink). ST is read directly off the
  realised trailing tape and is defined *relative to the market*, not a name's own third moment.

None of the siblings compute a **market-contrast salience ranking with declining decision weights**
— the Cosemans-Frehen / Bordalo-Gennaioli-Shleifer signal — which is this study's own axis.
