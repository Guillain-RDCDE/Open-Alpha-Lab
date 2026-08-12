# References & literature map — Study 883 (Mid-Cap Sweet Spot)

## The claim under test

- **The "forgotten middle" folklore.** A long-standing marketing and practitioner claim —
  repeated by S&P Dow Jones Indices, index-provider white papers and mid-cap ETF issuers —
  is that mid-cap stocks are a **risk-adjusted sweet spot**: large enough to be past the
  fragility and liquidity risk of micro-/small-caps, small enough to still grow faster than
  mega-caps, and *under-followed* relative to both (fewer analysts than large, more than
  small). The headline version says the mid-cap index has historically delivered a **higher
  Sharpe than both** the S&P 500 (large) and the Russell 2000 / S&P SmallCap (small).
- **What we actually test.** Whether the mid-cap ETF earns a genuine **excess-of-cash Sharpe
  advantage over BOTH** SPY and IWM that (a) is statistically distinguishable from zero, (b)
  holds across sub-eras, and (c) survives real costs. Every Sharpe is measured *excess of
  BIL cash* so the race is apples-to-apples; the pairwise return difference carries a
  Newey-West HAC *t*; a paired block bootstrap puts a CI on the Sharpe advantage.
- **S&P MidCap 400.** The mid-cap benchmark tracked by both IJH and MDY is the **S&P MidCap
  400**, launched by Standard & Poor's in 1991 — the 400 US companies in the mid-cap size
  band, float-weighted, with an explicit profitability screen. The Russell 2000 (IWM) and
  S&P 500 (SPY) bracket it on the small and large sides.

## What we measure, and the honesty rails

- **Excess-of-cash, always.** Each leg's Sharpe uses its daily return minus BIL's, so the
  ~4-5%/yr T-bill rate of 2023-26 does not flatter one leg over another (all four equity
  legs carry the same cash drag).
- **The cash leg dates the window.** BIL lists only from **2007-05**, which fixes the
  Sharpe-race common window at 2007-2026 — it **misses** the 1995-2006 stretch when mid-caps
  most out-ran large. This short-history limitation is named on the **Signal** axis. The
  pairwise return **difference** (mid − large) is cash-independent, so the era-robustness cut
  uses the full MDY tape back to 1995.
- **HAC inference.** Daily equity returns are mildly autocorrelated; the plain *t* overstates
  significance on an overlapping-window difference, so the pairwise difference and the costed
  spread carry a **Newey-West (Bartlett-kernel)** *t* with the standard `floor(4·(n/100)^{2/9})`
  lag rule.
- **A bootstrap on the advantage itself.** The Sharpe *advantage* (mid minus neighbour) is
  resampled with a **paired circular block bootstrap** (2,000 draws, 21-day blocks) that
  preserves the two legs' cross-correlation — a CI that spans zero means the advantage is not
  distinguishable from none.
- **Total return, one lag, costs separate.** Prices are `auto_adjust=True` total-return; the
  costed dollar-neutral spread charges one-way ETF spreads on both legs plus borrow on the
  short, graded on its own **Tradability** axis.
- **The synthetic control proves the machinery only.** A seeded world with a *planted* mid
  Sharpe edge (null at 0) shows the detector recovers a real advantage and stays quiet on the
  null — it never supports the real-tape stamp.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* on the difference and costed-spread series).
- **Politis, D. & Romano, J. (1994)** — the stationary / circular block bootstrap used for
  the paired Sharpe-advantage CI.
- **Lo, A. (2002)**, *"The Statistics of Sharpe Ratios"* (Financial Analysts Journal) — why a
  Sharpe ratio needs a standard error and how serial correlation inflates it.
- **Wilson, E. B. (1927)** — score interval for a binomial share.

## Data sources

- **yfinance daily closes** (`auto_adjust=True`, total-return): IJH, MDY, SPY, IWM, BIL,
  1993-01-29 → 2026-06-30, cached under this study's own `_cache/prices.parquet`.
- All headline numbers are pinned in [`results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [513-size-effect](../../513-size-effect/) — the classic **small-minus-big** (SMB) size
  premium: small-caps out-earn large-caps. This study is *not* about small-vs-large; it asks
  whether the **middle** beats **both** ends, a distinct (and non-monotone) claim.
- [177-megacap-concentration](../../177-megacap-concentration/) — the top-heaviness of the
  **large-cap** index (a handful of mega-caps dominating the S&P 500). This study sorts by
  the **size band** of the whole fund, not intra-index concentration.
- [94-level-pegging](../../94-level-pegging/) — **equal-weighting** an index to lean away from
  mega-caps. That re-weights *within* one universe; here we compare three *separate* size-band
  funds head-to-head.
- [657-larry-portfolio](../../657-larry-portfolio/) — the **small-value** tilt (small-cap ×
  value factor). This study takes plain-vanilla **mid-cap blend** exposure, no value screen,
  and pits it against large and small blend.

None of the siblings test the **mid-cap band's excess-of-cash Sharpe against BOTH neighbours
at once** — the "forgotten middle" sweet-spot claim — which is this study's own axis.
