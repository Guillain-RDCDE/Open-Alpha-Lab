# References & literature map — Study 803 (Realized-Skewness Reversal)

## The claim under test

- **The source paper.** Diego **Amaya, Peter Christoffersen, Kris Jacobs & Aurelio
  Vasquez**, *"Does Realized Skewness Predict the Cross-Section of Equity Returns?"*
  (Journal of Financial Economics, 2015). Building weekly **realized** moments from
  high-frequency returns, they find that a stock's **realized skewness** has a robust
  **negative** relation to its next-week return: the names with the most right-skewed
  recent return distribution go on to under-earn, and a long low-skew / short high-skew
  portfolio earns a positive spread. Realized **kurtosis** is weakly positive; realized
  **volatility** is the familiar negative low-risk tilt.
- **The behavioural reading.** A right-skewed (lottery-like) recent tape attracts
  skewness-loving investors who bid the name up, lowering its subsequent return —
  the same over-pricing-of-the-tail mechanism behind the MAX and expected-idiosyncratic-
  skewness effects, but measured directly ex-post from the realized third moment.
- **The specific test here.** We take the self-contained daily version: sort a liquid
  US cross-section on its **trailing realized skewness of daily returns** and measure
  the forward return of the equal-weight long-low-skew / short-high-skew book, with a
  Newey-West *t*, a permutation placebo, a two-era robustness cut, a costed timer, and a
  seeded synthetic positive control. (Daily returns are a coarser skewness estimator
  than the paper's 5-minute intraday sampling, so the magnitudes here are conservative.)

## What we measure, and the honesty rails

- **Realized skewness, no free model.** For each name, the rolling `window`-day sample
  skewness of daily simple returns (population third standardised moment), computed
  vectorised via the moment identity `m3 / m2**1.5`.
- **Point-in-time sort, one documented lag.** The ranking signal is the trailing
  skewness **known at the close of `t-1`** (`.shift(1)`); the book is held on day `t`.
  Zero look-ahead.
- **Robust inference.** Newey-West (HAC, Bartlett, 10-lag) *t* on the daily long-short
  spread — an overlapping-formation signal is serially correlated, so a plain *t* would
  overstate significance. A one-sample *t* and a pooled Welch *t* (low-skew book vs
  high-skew book) cross-check. A **1,000-permutation placebo** breaks the
  signal → forward-return link to confirm the spread isn't a lucky alignment of the sort.
- **Survivorship is named on the Signal axis.** The universe is a **current-membership**
  set of ~50 liquid mega-caps, pulled through the `quantlab.universe` survivorship guard
  (`allow_survivorship_bias=True`, an explicit opt-in). Delisted / de-rated names are
  absent, so the cross-sectional magnitudes are an **upper bound**.
- **The timer is graded separately.** Costs are one-way × NAV on the long-short book,
  and the short book pays borrow — the honest test of whether a small daily spread
  survives friction.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the spread series).
- **Wilson, E. B. (1927)** — score interval for a binomial share.
- **Boyer, B., Mitton, T. & Vorkink, K. (2010)** — expected idiosyncratic skewness (the
  *ex-ante* cousin tested in study 503; this study uses the *realized* moment instead).

## Data sources

- **yfinance daily OHLC** (`auto_adjust=True`, total-return), 50 liquid US large-caps,
  2010-01-04 → 2026-06-30, cached under `_cache/` through `quantlab.universe`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [503-expected-idiosyncratic-skewness](../../503-expected-idiosyncratic-skewness/) —
  the **ex-ante / expected** idiosyncratic skewness (Boyer-Mitton-Vorkink), a *modelled
  forecast* of skewness. This study uses the **realized** (ex-post) third moment read
  directly off the return tape — a different signal, a different paper.
- [504-coskewness](../../504-coskewness/) — **systematic** co-skewness with the market
  (how a stock's return co-moves with market variance), a beta-like exposure. This study
  sorts on a name's **own** total realized skewness, not its co-movement.
- [365-lottery-max-effect](../../365-lottery-max-effect/) — the single **maximum daily
  return** (MAX) over a month, a one-number tail proxy. Realized skewness is the full
  third moment of the distribution, not the extreme order statistic.

None of the siblings sort on the **realized third moment of a name's own daily returns**
— the Amaya-et-al signal — which is this study's own axis.
