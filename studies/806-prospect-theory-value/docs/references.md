# References & literature map — Study 806 (Prospect-Theory Value)

## The claim under test

- **The source paper.** Nicholas **Barberis, Abhiroop Mukherjee & Baolian Wang**,
  *"Prospect Theory and Stock Returns: An Empirical Test"* (Review of Financial
  Studies, 2016). For each stock they compute the **cumulative-prospect-theory (TK)
  value** a Tversky-Kahneman investor would assign to the stock's *past* return
  distribution (the previous 60 monthly returns), and find it predicts the
  cross-section of future returns **negatively**: stocks whose recent return
  distribution looks like an attractive gamble (a **high** TK value) go on to earn
  **lower** returns. The mechanism is the *narrow framing* of a single stock as a
  standalone gamble — investors evaluate it through prospect theory, overweight the
  good gamble, over-pay, and it under-performs.
- **The prospect-theory engine.** Amos **Tversky & Daniel Kahneman**, *"Advances in
  Prospect Theory: Cumulative Representation of Uncertainty"* (Journal of Risk and
  Uncertainty, 1992). The value function `v(x)=x^0.88` for gains, `-2.25*(-x)^0.88`
  for losses (loss aversion `λ=2.25`), and the inverse-S probability-weighting
  functions `w+` (`γ=0.61`) and `w-` (`δ=0.69`) that overweight the tails. These are
  the exact parameters used here.
- **Original prospect theory.** Daniel **Kahneman & Amos Tversky**, *"Prospect
  Theory: An Analysis of Decision under Risk"* (Econometrica, 1979) — the
  reference-dependence, loss-aversion and probability-distortion foundations.
- **The behavioural reading.** A right-skewed, lottery-like recent tape puts mass in
  the probability-overweighted right tail, so its TK value is high; prospect-theory
  investors bid it up and its subsequent return is low — the same over-pricing-of-the-
  upside mechanism behind the MAX and expected-idiosyncratic-skewness effects, but
  captured by the *full* TK functional (value function **and** probability weighting),
  not a single moment.
- **The specific test here.** We take a self-contained daily version: for each name we
  form the empirical distribution from its **trailing ~5 years (1,260 trading days) of
  daily returns**, compute its TK value, sort a liquid US cross-section monthly, and
  measure the forward one-month return of the equal-weight long-low-TK / short-high-TK
  book, with a Newey-West *t*, a permutation placebo, a two-era robustness cut, a
  costed timer, and a seeded synthetic positive control. (Daily outcomes over 5y are a
  richer estimate of the gamble than the paper's 60 monthly returns.)

## What we measure, and the honesty rails

- **TK value, exact Tversky-Kahneman parameters.** Sort each name's trailing daily
  returns, treat each as equally likely, apply `v(x)` and the rank-dependent decision
  weights (successive differences of the tail-cumulated `w+`/`w-`), and sum. The
  decision weights are deliberately **subadditive** (they do not sum to one — the CPT
  subcertainty feature), a property the test-suite pins.
- **Point-in-time monthly sort, one documented lag.** The ranking signal is the TK
  value **known at the close of month `t`**; the book is held over month `t+1`. A name
  is eligible only once a **full** trailing window is available (no truncated early
  windows). Zero look-ahead.
- **Robust inference.** Newey-West (HAC, Bartlett, 6-lag) *t* on the monthly long-short
  spread. A one-sample *t* and a pooled Welch *t* (low-TK book vs high-TK book)
  cross-check. A **1,000-permutation placebo** breaks the signal → forward-return link
  to confirm the spread isn't a lucky alignment of the sort.
- **Survivorship is named on the Signal axis.** The universe is a **current-membership**
  set of ~50 liquid mega-caps, pulled through the `quantlab.universe` survivorship guard
  (`allow_survivorship_bias=True`, an explicit opt-in). Delisted / de-rated names are
  absent — most bitingly on the *short* leg (blown-up lottery names never enter) — so
  the cross-sectional magnitudes are an **upper bound**.
- **The timer is graded separately.** Costs are 2 sides × one-way × NAV per monthly
  rebalance, and the short book pays borrow — the honest test of whether the spread
  survives friction *and* the survivorship caveat.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the monthly spread series).
- **Wilson, E. B. (1927)** — score interval for a binomial share.
- **Boyer, B., Mitton, T. & Vorkink, K. (2010)** — expected idiosyncratic skewness (the
  *ex-ante modelled* cousin tested in study 503; this study uses the full TK functional
  instead of a single skewness forecast).

## Data sources

- **yfinance daily OHLC** (`auto_adjust=True`, total-return), 50 liquid US large-caps,
  2010-01-04 → 2026-06-30, cached under `_cache/` through `quantlab.universe`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [365-lottery-max-effect](../../365-lottery-max-effect/) — the single **maximum daily
  return** (MAX) over a month, a one-number tail proxy (Bali-Cakici-Whitelaw). This
  study uses the **whole probability-weighted TK value** of the return distribution,
  not one extreme order statistic.
- [327-disposition-overhang](../../327-disposition-overhang/) — the **capital-gains
  overhang** driving disposition-effect selling pressure, a reference-point *holding*
  story (unrealised gains/losses relative to cost basis). This study prices the past
  *return distribution* as a standalone **gamble**, not the position's embedded P&L.
- [503-expected-idiosyncratic-skewness](../../503-expected-idiosyncratic-skewness/) —
  the **ex-ante / modelled** idiosyncratic skewness (Boyer-Mitton-Vorkink), a single
  moment forecast. This study applies the **full Tversky-Kahneman functional** (value
  function + inverse-S probability weighting) to the realised distribution.

None of the siblings compute the **cumulative-prospect-theory value of a name's own
past return distribution** — the Barberis-Mukherjee-Wang signal — which is this study's
own axis.
