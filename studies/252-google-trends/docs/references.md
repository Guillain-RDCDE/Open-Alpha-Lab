# References & literature map -- Study 252 (Search-Trends)

## The canonical claim

- **Da, Z., Engelberg, J. & Gao, P. (2011).** *In Search of Attention.* Journal
  of Finance, 66(5), 1461--1499.  The founding paper: the Google Search Volume
  Index (SVI) is a direct, timely proxy for *retail* investor attention.  An
  abnormal jump in SVI predicts higher prices over the next two weeks and a
  **price reversal** over the following year -- i.e. attention is uninformed
  buying pressure, not information.  Strongest for small, retail-held Russell
  3000 names.  Our study tests whether this reversal survives a *monthly,
  mega-cap, proxy-Trends* implementation (it does not).

- **Da, Z., Engelberg, J. & Gao, P. (2015).** *The Sum of All FEARS: Investor
  Sentiment and Asset Prices.* Review of Financial Studies, 28(1), 1--32.
  Aggregates household search queries into a sentiment index (FEARS) that
  predicts short-term return reversals and volatility -- reinforcing the
  "attention = transient pressure" reading.

## Supporting and competing evidence

- **Bank, M., Larch, M. & Peter, G. (2011).** *Google Search Volume and its
  Influence on Liquidity and Returns of German Stocks.* Financial Markets and
  Portfolio Management, 25(3), 239--264.  Confirms increased SVI raises trading
  activity, liquidity, and short-horizon returns in DAX names.

- **Preis, T., Moat, H. S. & Stanley, H. E. (2013).** *Quantifying Trading
  Behavior in Financial Markets Using Google Trends.* Scientific Reports, 3,
  1684.  A widely cited (and widely critiqued) claim that a Trends-based
  "debt"-keyword strategy would have beaten buy-and-hold -- later shown to be
  fragile to keyword choice and sample period (a cautionary tale on Trends
  data-mining).

- **Vlastakis, N. & Markellos, R. N. (2012).** *Information Demand and Stock
  Market Volatility.* Journal of Banking & Finance, 36(6), 1808--1821.  Search
  intensity co-moves with volatility and volume more than with returns --
  attention is a risk/turnover signal, not a clean alpha signal.

## Why our result is negative

- **McLean, R. D. & Pontiff, J. (2016).** *Does Academic Research Destroy Stock
  Return Predictability?* Journal of Finance, 71(1), 5--32.  Anomaly returns
  decay ~58% post-publication; the attention effect (published 2011) has been
  heavily arbitraged, and our most-recent sub-period leans the opposite way
  (momentum, t = -3.78) rather than reversal.

- **Hand-curated-proxy caveat.** Google Trends exposes only quantised 0-100
  *relative* indices with no stable free bulk API; our attention table is a
  deterministic, hand-parameterised *proxy* of the public Trends shape
  (holiday/launch/event spikes), not a tick-exact pull.  This is named on the
  Signal axis: a real SVI pull might differ, but the horizon (monthly) and
  universe (30 mega-cap survivors) are the binding constraints, not the proxy.

## Related desk studies

- **[Study 223 -- Same-Month-Seasonality](../../223-same-month-seasonality/)**:
  the reference pattern for this study (synthetic panel + cached real proxy +
  decile/tercile cross-sectional sort).
- **[Study 158 -- Super-Bowl](../../158-super-bowl/)**: the hardcoded-table
  pattern for an offline, deterministic curated data series.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
  Matrix* (Econometrica).
- **Block-bootstrap Sharpe CI.** Politis & Romano (1994), *The Stationary
  Bootstrap* (JASA).
- **Survivorship notation.** Shumway (1997), *The Delisting Bias in CRSP Data*
  (Journal of Finance).
