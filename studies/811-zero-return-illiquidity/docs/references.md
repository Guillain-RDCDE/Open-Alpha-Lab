# References & literature map — Study 811 (Zero-Return Illiquidity)

## The claim under test

- **The source paper.** David A. **Lesmond, Joseph P. Ogden & Charles A. Trzcinka**,
  *"A New Estimate of Transaction Costs"* (Review of Financial Studies, 1999). Their
  insight: when the round-trip cost of trading a security exceeds the value of the
  information arriving that day, the informed trader stays out and the security's price
  **does not move** — so the observed daily return is **exactly zero**. The *frequency
  of zero-return days* therefore proxies a name's total (explicit + implicit)
  transaction cost. The full LOT estimator fits a limited-dependent-variable (Tobit-like)
  model to recover an implied round-trip cost; the **proportion of zero-return days** is
  its cheap, model-free, price-only reduced form — the version tested here.
- **Why an illiquidity *premium*.** **Amihud & Mendelson (1986)**, *"Asset Pricing and
  the Bid-Ask Spread"*: investors must be compensated for holding assets that are costly
  to trade, so higher-illiquidity names carry higher *expected* returns. If the
  zero-return proportion is a valid illiquidity proxy, sorting on it and going long the
  high-zero (illiquid) names should earn a positive spread.
- **The specific test here.** We take the reduced-form daily version: each name's
  **trailing-252-day proportion of exactly-zero (`|r| < 1e-8`) daily returns**, sorted
  point-in-time, long the top 30% / short the bottom 30%, with a Newey-West *t*, a
  permutation placebo, a two-era robustness cut, a costed timer, and a seeded synthetic
  positive control. Total-return (`auto_adjust`) closes are used, so genuine no-change
  days — not dividend artefacts — drive the zeros.

## The honest wrinkle, stated up front

- **Zero-return days need a coarse price grid.** The LOT zeros arise on tick-priced,
  thinly-traded names where the minimum price increment and the no-trade condition make
  the close repeat. Liquid mega-caps, priced in cents on billions of dollars of volume,
  **almost never** print an exactly-zero adjusted return: on this panel the median name's
  trailing-year zero-proportion is **0.00%** and the maximum is **2.38%**. The signal is
  therefore **near-degenerate** on a mega-cap universe — the regime where an illiquidity
  premium is least likely to exist — so a **None** stamp is the *expected*, not the
  surprising, outcome. We report it plainly.

## What we measure, and the honesty rails

- **Zero-return proportion, no free model.** For each name, the rolling `window`-day mean
  of the exact-zero indicator (`|r| < 1e-8`), computed vectorised as a rolling mean.
- **Point-in-time sort, one documented lag.** The ranking signal is the trailing zero
  proportion **known at the close of `t-1`** (`.shift(1)`); the book is held on day `t`.
  Zero look-ahead.
- **Robust inference.** Newey-West (HAC, Bartlett, 10-lag) *t* on the daily long-short
  spread — an overlapping-formation signal is serially correlated, so a plain *t* would
  overstate significance. A one-sample *t* and a pooled Welch *t* (illiquid book vs liquid
  book) cross-check. A **1,000-permutation placebo** breaks the signal → forward-return
  link to confirm the (small) spread isn't a lucky alignment of the sort.
- **Survivorship is named on the Signal axis.** The universe is a **current-membership**
  set of ~50 liquid mega-caps, pulled through the `quantlab.universe` survivorship guard
  (`allow_survivorship_bias=True`, an explicit opt-in). Delisted / de-rated names are
  absent, so the cross-sectional magnitudes are an **upper bound**.
- **The timer is graded separately.** Costs are one-way × NAV on the long-short book, and
  the short book pays borrow — and the long leg is by construction the least-liquid names,
  so the charged cost is an optimistic floor.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the spread series).
- **Wilson, E. B. (1927)** — score interval for a binomial share.
- **Fong, K., Holden, C. & Trzcinka, C. (2017)** — a modern horse-race of low-frequency
  liquidity proxies (zero-return, Amihud, high-low spread) against high-frequency
  benchmarks; documents that the zero-return measure is informative mainly in illiquid
  cross-sections.

## Data sources

- **yfinance daily OHLC** (`auto_adjust=True`, total-return), 50 liquid US large-caps,
  2010-01-04 → 2026-06-30, cached under `_cache/` through `quantlab.universe`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [140-amihud-illiquidity](../../140-amihud-illiquidity/) — the **Amihud ILLIQ** ratio
  (average |return| per dollar of volume), a *volume-scaled price-impact* measure. The
  zero-return proportion uses **no volume at all**: it is the LOT *price-only* proxy, a
  different construction of "illiquidity."
- [141-turnover](../../141-turnover/) — share **turnover** (volume / shares outstanding),
  a raw *trading-activity* measure. Zero-return frequency is a *cost* proxy read off the
  price path, not an activity statistic — and again needs no volume.
- [812-corwin-schultz](../../812-corwin-schultz/) — the Corwin-Schultz **high-low spread**
  estimator, which backs an implied bid-ask spread out of two-day high/low ranges. This
  study uses only the close-to-close return and its exact-zero frequency, not the intraday
  range.

None of the siblings sort on the **frequency of exactly-zero close-to-close returns** —
the Lesmond-Ogden-Trzcinka signal — which is this study's own axis.
