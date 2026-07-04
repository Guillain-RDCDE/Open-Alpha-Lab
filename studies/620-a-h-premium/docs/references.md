# References — Study 620 (A-H Share Premium)

## The claim's source

- **Hang Seng Stock Connect China AH Premium Index (HSAHP)** — the industry's own daily
  measure of the FX-adjusted A-over-H price gap for dual-listed Chinese companies; launched
  2007, base 100 = parity. Our per-pair construction (raw A close ÷ (raw H close × HKD→CNY))
  is the same convention, pair by pair.
  <https://www.hsi.com.hk/eng/indexes/all-indexes/ahpremium>
- The "30% more in Shanghai" folklore is the HSAHP hovering around 130–150 through the
  2015–2024 decade, quoted in essentially every China market note. Our 14-pair equal-weight
  panel averages **+29.7%** over 2010–2026 — the folklore number is the tape's number.

## Why the premium can exist at all (the no-arbitrage-channel literature)

- Bailey, W. (1994). *Risk and return on China's new stock markets: Some preliminary
  evidence.* Pacific-Basin Finance Journal 2(2–3), 243–260 — the first documentation of the
  (then) foreign-share discount. <https://doi.org/10.1016/0927-538X(94)90016-7>
- Fernald, J. & Rogers, J. H. (2002). *Puzzles in the Chinese stock market.* Review of
  Economics and Statistics 84(3), 416–432 — segmentation + differing required returns as the
  premium's engine. <https://doi.org/10.1162/003465302320259420>
- Chan, K., Menkveld, A. J. & Yang, Z. (2008). *Information asymmetry and asset prices:
  Evidence from the China foreign share discount.* Journal of Finance 63(1), 159–196.
  <https://doi.org/10.1111/j.1540-6261.2008.01313.x>
- Mei, J., Scheinkman, J. A. & Xiong, W. (2009). *Speculative trading and stock prices:
  Evidence from Chinese A-B share premia.* Annals of Economics and Finance 10(2), 225–255 —
  speculative/turnover demand priced into the mainland line.
  <https://www.princeton.edu/~wxiong/papers/ChinaAB.pdf>

The structural facts that close every arbitrage: **A and H lines are non-fungible** (one
cannot be converted into the other); **A-shares are effectively unborrowable** for outside
investors (no securities lending through Stock Connect; the CSRC suspended the domestic
securities *relending* channel in July 2024 — see Reuters,
<https://www.reuters.com/markets/asia/>); mainland capital controls segment the investor
bases; and dividend withholding differs by channel (10% on northbound A-share dividends via
Connect; 20% for mainland investors holding H southbound — HKEX Stock Connect rules,
<https://www.hkex.com.hk/Mutual-Market/Stock-Connect>).

- **Shanghai-Hong Kong Stock Connect** launched 2014-11-17 (Shenzhen leg 2016-12-05) — the
  event that let the same investors *buy* both lines without letting anyone *arb* them; our
  post-Connect subsample starts 2014-12. <https://www.hkex.com.hk/Mutual-Market/Stock-Connect>

## Method

- Newey, W. K. & West, K. D. (1987). *A simple, positive semi-definite, heteroskedasticity
  and autocorrelation consistent covariance matrix.* Econometrica 55(3), 703–708 — the HAC
  *t* used throughout. <https://doi.org/10.2307/1913610>
- Dickey, D. A. & Fuller, W. A. (1979). *Distribution of the estimators for autoregressive
  time series with a unit root.* JASA 74, 427–431 — the −2.86 critical value in the
  non-convergence test. <https://doi.org/10.2307/2286348>
- The near-unit-root honesty device (Monte-Carlo null calibrated to the tape's own AR(1)
  instead of trusting an HAC *t* on a level) follows the spirit of the desk's
  [METHODOLOGY](../../../METHODOLOGY.md) inference bar; our own synthetic control shows 26/40
  zero-mean worlds faking |HAC *t*| ≥ 2 on the level.

## Named siblings (dedup guard)

- [05-twin-spread](../../05-twin-spread/) is **Gatev-style distance pairs**: statistically
  co-moving *different* companies, betting the spread's historical distance re-closes. This
  study is the **structural dual-listing premium**: the *same* company, two non-fungible
  tickers, a premium with **no convergence mechanism at all** — the opposite end of the
  pairs-trading spectrum (there, reversion is the whole trade; here, the level is
  statistically indistinguishable from a random walk and only the *cross-section* reverts).
- [618-gbtc-premium-cycle](../../618-gbtc-premium-cycle/) is the same *wrapper-premium*
  species in a single US vehicle, where a creation/redemption channel eventually opened and
  killed the premium — the control experiment for what a real arb channel does. Here the
  channel never opens.

## Data

- Yahoo Finance daily closes (raw + dividend-adjusted), no key: 14 A-share lines
  (`601318.SS`, `601857.SS`, `600028.SS`, `601398.SS`, `601988.SS`, `601939.SS`,
  `601288.SS`, `601628.SS`, `600036.SS`, `600030.SS`, `601088.SS`, `601111.SS`,
  `600600.SS`, `601633.SS`), their H twins (`2318.HK`, `0857.HK`, `0386.HK`, `1398.HK`,
  `3988.HK`, `0939.HK`, `1288.HK`, `2628.HK`, `3968.HK`, `6030.HK`, `1088.HK`, `0753.HK`,
  `0168.HK`, `2333.HK`) and the FX crosses `CNY=X`, `HKD=X`. <https://finance.yahoo.com/>
- Data hygiene: Yahoo backfills pre-listing placeholder prices of 0.01 on some HK lines
  (6030.HK before 2011-10-06); closes ≤ 0.05 are masked in the loader.
- Cache: [`_cache/ahp_close.csv`](../_cache/ahp_close.csv),
  [`_cache/ahp_adj.csv`](../_cache/ahp_adj.csv); everything runs cache-first and offline.
