# References & literature map — Study 812 (Corwin-Schultz Spread)

## The claim under test

- **The source paper.** Shane A. **Corwin & Paul Schultz**, *"A Simple Way to Estimate
  Bid-Ask Spreads from Daily High and Low Prices"* (Journal of Finance, 2012, 67(2):
  719–759). Their insight: the daily **high** almost always transacts at (or near) the
  **ask** and the daily **low** at the **bid**, so the high-low *ratio* embeds the spread —
  while the price *variance* over an interval scales with time but the spread does not.
  Comparing single-day squared log-ranges (`beta`) with the two-day log-range (`gamma`)
  isolates the spread: `alpha = (√(2β)−√β)/(3−2√2) − √(γ/(3−2√2))` and the proportional
  spread is `S = 2(e^α−1)/(1+e^α)`, with negative daily estimates floored at 0.
- **The economic reading tested here.** A high estimated spread is a proxy for
  **illiquidity**; illiquid assets must offer higher expected returns to clear
  (Amihud-Mendelson). So sorting the cross-section on the Corwin-Schultz spread and buying
  the illiquid (high-spread) names against the liquid (low-spread) names is the classic
  **illiquidity-premium** bet: high-spread names should out-earn.
- **The specific test here.** Daily Corwin-Schultz `S` per name, averaged over a trailing
  month, then a point-in-time equal-weight long-top-30% / short-bottom-30% sort, graded with
  a Newey-West *t*, a permutation placebo, a two-era robustness cut, a costed timer, and a
  seeded synthetic positive control.

## What we measure, and the honesty rails

- **The estimator, no free model.** Per name, the daily two-day `beta`/`gamma`/`alpha`/`S`
  recursion above (vectorised over the whole panel), negatives floored at 0, then a
  trailing-21-day mean as the estimated spread. The median mega-cap estimate (~13 bps) is a
  sensible effective spread — a sanity check on the estimator itself.
- **Point-in-time sort, one documented lag.** The ranking spread is **known at the close of
  `t-1`** (`.shift(1)`); the book is held on day `t`. Zero look-ahead.
- **Robust inference.** Newey-West (HAC, Bartlett, 10-lag) *t* on the daily long-short
  spread — an overlapping trailing-window signal is serially correlated, so a plain *t* would
  overstate significance. A one-sample *t* and a pooled Welch *t* (high-spread vs low-spread
  book) cross-check. A **1,000-permutation placebo** breaks the signal → forward-return link.
- **Survivorship is named on the Signal axis.** The universe is a **current-membership** set
  of ~50 liquid mega-caps, pulled through the `quantlab.universe` survivorship guard
  (`allow_survivorship_bias=True`, an explicit opt-in). Delisted / de-rated names are absent,
  so the cross-sectional magnitudes are an **upper bound**.
- **The timer is graded separately.** Costs are one-way × NAV on the long-short book, and the
  short book pays borrow — the honest test of whether the premium survives friction, and note
  that the *long* leg is the illiquid names where real spreads are widest.

## Shared method citations

- **Amihud, Y. & Mendelson, H. (1986)** — asset pricing and the bid-ask spread; the
  theoretical basis for an illiquidity premium (illiquid assets earn more).
- **Amihud, Y. (2002)** — illiquidity and stock returns; the |return|/volume price-impact
  proxy (the volume-based cousin tested in study 140).
- **Roll, R. (1984)** — an earlier serial-covariance estimator of the effective spread that
  Corwin-Schultz improves on (it can imply negative variances; the high-low estimator is more
  robust).
- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the spread series).
- **Wilson, E. B. (1927)** — score interval for a binomial share.

## Data sources

- **yfinance daily OHLC** (`auto_adjust=True`, total-return), 50 liquid US large-caps,
  2010-01-04 → 2026-06-30, cached under `_cache/` through `quantlab.universe`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [377-bid-ask-bounce](../../377-bid-ask-bounce/) — the short-horizon **mean-reversion**
  induced by the spread's *bounce* (a return-autocorrelation timing signal on a single name).
  This study uses the spread's estimated **level** as a cross-sectional illiquidity sort, not
  its serial-correlation footprint.
- [140-amihud-illiquidity](../../140-amihud-illiquidity/) — Amihud's
  |return| / dollar-volume **price-impact** ratio, a *volume*-based illiquidity proxy.
  Corwin-Schultz needs **no volume at all** — it reads the spread straight off the high-low
  range, a different construction of the same latent illiquidity.
- [811-zero-return-days](../../811-zero-return-days/) — the **count of zero-return days**
  (Lesmond-Ogden-Trzcinka) as an illiquidity proxy, a *frequency* measure. This study uses the
  **high-low range** magnitude, not the incidence of no-trade days.

None of the siblings estimate the spread from the **daily high-low range** the way
Corwin-Schultz does — this study's own axis. All three are alternative illiquidity proxies;
they can and do disagree name-by-name.
