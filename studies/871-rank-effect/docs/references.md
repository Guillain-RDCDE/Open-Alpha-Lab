# References & literature map — Study 871 (The Rank Effect)

## The claim under test

- **The source paper.** Samuel M. **Hartzmark**, *"The Worst, the Best, Ignoring All the
  Rest: The Rank Effect and Trading Behavior"* (Review of Financial Studies, 2015).
  Studying individual and institutional portfolios, Hartzmark documents the **rank
  effect**: an investor is far more likely to **sell a position ranked at the extreme**
  of her portfolio — the **best** performer and the **worst** performer — than a
  middle-ranked position, even after controlling for the raw return of the position. The
  *ordinal rank* itself, a salience heuristic, drives the trade.
- **The behavioural reading.** Extreme ranks are **salient**: the top and bottom of a
  sorted list draw attention (an edge / anchoring effect), so investors act on them
  disproportionately. This is a **rank-position** phenomenon, deliberately separated in
  the paper from the level of the return, from the disposition effect, and from
  portfolio-weight effects.
- **The specific test here.** We take a self-contained cross-sectional proxy: each day
  **rank a liquid US cross-section by trailing return**, and ask whether the
  **extreme-ranked** names (rank 1 and rank N) go on to **underperform the middle** —
  the predictable-selling-pressure prediction — with the raw return level explicitly
  controlled. If the extremes are being sold disproportionately, a long-middle /
  short-extremes book should earn a positive spread. We stamp it with a Newey-West *t*, a
  level-controlled (residualised) cut, a permutation placebo, a two-era robustness cut, a
  costed timer, and a seeded synthetic positive control. (This is a *return-predictability*
  proxy for a *trading-behaviour* result; the mapping from selling pressure to realised
  cross-sectional returns is exactly what the test interrogates.)

## What we measure, and the honesty rails

- **Rank extremity, no free model.** For each name, the rolling 42-day trailing return;
  each day the cross-section is ranked and a name's **extremity** `|2u − 1| ∈ [0,1]`
  (fractional rank `u`) is 0 in the middle and 1 at either tail. The tradable book longs
  the middle band and shorts both tails, equal weight.
- **Point-in-time sort, one documented lag.** The ranking signal is the trailing return
  **known at the close of `t−1`** (`.shift(1)`); the book is held on day `t`. Zero
  look-ahead.
- **Controlling for the raw return level.** The both-tails short is *approximately*
  level-neutral by construction (a rank-1 winner and a rank-N loser average toward the
  middle's level — see the `lvl_mid` vs `lvl_ext` diagnostic). The explicit control
  residualises each day's forward return on a **quadratic in the standardised
  trailing-return level** before re-measuring the spread, so *any* smooth momentum /
  reversal curve is removed and only the rank-*position* effect survives.
- **Robust inference.** Newey-West (HAC, Bartlett, 10-lag) *t* on the daily long-short
  spread — an overlapping-formation signal is serially correlated, so a plain *t* would
  overstate significance. A one-sample *t* and a pooled Welch *t* (middle book vs
  extremes book) cross-check. A **1,000-permutation placebo** breaks the
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
- **Fama, E. & MacBeth, J. (1973)** — the cross-sectional-regression-per-period idea
  behind the level-controlled residualisation.
- **Wilson, E. B. (1927)** — score interval for a binomial share.

## Data sources

- **yfinance daily OHLC** (`auto_adjust=True`, total-return), 50 liquid US large-caps,
  2010-01-04 → 2026-06-30, cached under `_cache/` through `quantlab.universe`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [327-disposition-effect](../../327-disposition-effect/) — the tendency to **sell
  winners and ride losers** relative to a *purchase-price* reference point. The rank
  effect is **reference-free** and **symmetric**: it is the extreme *rank position*
  (top *and* bottom of the portfolio) that drives selling, not the sign of the gain.
- [365-lottery-max-effect](../../365-lottery-max-effect/) — sorts on the single
  **maximum daily return** (a right-tail lottery proxy). This study uses a name's
  **rank extremity** among its peers, symmetric across *both* tails, not an extreme
  order statistic.
- [806-prospect-theory-value](../../806-prospect-theory-value/) — a prospect-theory
  **value** of the whole gain/loss distribution (a valuation of the return path). The
  rank effect uses only the **ordinal position** of a name in the cross-section.
- [202-fifty-two-week-low](../../202-fifty-two-week-low/) — nearness to a **52-week
  extreme price**, an anchor relative to a name's *own* price history. The rank effect is
  a **cross-sectional** rank among peers *this* period, not a self-history anchor.

None of the siblings sort on a name's **cross-sectional rank extremity** — top-*and*-bottom
of the trailing-return ranking, level-controlled — which is this study's own axis.
