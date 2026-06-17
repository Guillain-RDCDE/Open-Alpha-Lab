# References & literature map -- Study 253 (Wiki-Views)

## The canonical claim

- **Moat, H. S., Curme, C., Avakian, A., Kenett, D. Y., Stanley, H. E. &
  Preis, T. (2013).** *Quantifying Wikipedia Usage Patterns Before Stock Market
  Moves.* Scientific Reports, 3, 1801.
  The founding "Wikipedia views predict markets" paper: changes in the number of
  views of Wikipedia pages relating to companies in the DJIA were associated with
  subsequent market moves; a strategy that traded on increases in views was
  reported to have outperformed. Heavily caveated by later replication failures
  (small sample, many degrees of freedom, in-sample selection of pages).

- **Da, Z., Engelberg, J. & Gao, P. (2011).** *In Search of Attention.* Journal
  of Finance, 66(5), 1461--1499.
  The attention-proxy benchmark: Google Search Volume Index (SVI) as a direct
  measure of retail attention. High abnormal attention predicts higher prices
  over the next two weeks and an eventual *reversal* within the year -- the
  reversal half is the basis for our "short the surge, long the drought" bet.

## Mechanisms, proxies and skepticism

- **Preis, T., Moat, H. S. & Stanley, H. E. (2013).** *Quantifying Trading
  Behavior in Financial Markets Using Google Trends.* Scientific Reports, 3, 1684.
  The much-cited "Google Trends beats the market" result; later shown to be
  fragile to look-ahead in the rolling-window definition and to the choice of
  search terms -- a cautionary tale for any attention-signal study.

- **Barber, B. M. & Odean, T. (2008).** *All That Glitters: The Effect of
  Attention and News on the Buying Behavior of Individual and Institutional
  Investors.* Review of Financial Studies, 21(2), 785--818.
  Attention-grabbing stocks are net bought by retail investors, pushing prices up
  temporarily before reversal -- the behavioural mechanism behind the prior.

- **Vlastakis, N. & Markellos, R. N. (2012).** *Information Demand and Stock
  Market Volatility.* Journal of Banking & Finance, 36(6), 1808--1821.
  Information demand (search intensity) is tied to volatility and trading volume
  more than to a clean directional return signal -- consistent with our null.

- **Bartov, E., Faurel, L. & Mohanram, P. (2018).** *Can Twitter Help Predict
  Firm-Level Earnings and Stock Returns?* The Accounting Review, 93(3), 25--57.
  A reminder that social/attention signals that "work" are usually
  earnings-window and intraday-to-weekly, not a monthly cross-sectional spread.

## Why monthly mega-cap page-views are the wrong frequency

The published attention effects are *short-horizon* (days to two weeks) and
strongest in *small, illiquid, retail-heavy* names. A monthly rebalance on a
liquid mega-cap basket -- the only universe with clean, continuous Wikipedia
articles -- averages the effect away and is exactly where one expects a null.
Our result is consistent with this: the curated monthly surge carries no forward
information at the monthly horizon.

## Related desk studies

- **[Study 223 -- Same-Month Seasonality](../../223-same-month-seasonality/)**:
  the cross-sectional decile-sort machinery (HAC t, random-portfolio control,
  turnover drag) reused here.
- **[Study 158 -- Super-Bowl](../../158-super-bowl/)**: the curated hardcoded
  event/series-table pattern (offline, deterministic) reused for the page-view
  anchors.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
  Matrix* (Econometrica).
- **Block-bootstrap Sharpe CI.** Politis & Romano (1994), *The Stationary
  Bootstrap* (JASA).
- **Survivorship notation.** Shumway (1997), *The Delisting Bias in CRSP Data*
  (Journal of Finance).
