# References & literature map -- Study 258 (Baker-Wurgler)

## The canonical claim

- **Baker, M. & Wurgler, J. (2006).** *Investor Sentiment and the Cross-Section of
  Stock Returns.* Journal of Finance, 61(4), 1645--1680.
  The founding paper. Builds a composite sentiment index from six proxies (the
  closed-end-fund discount, NYSE turnover, the number and first-day returns of IPOs,
  the equity share in new issues, and the dividend premium), orthogonalized to
  macro fundamentals. Finds that when sentiment is *high*, subsequent returns are
  *low* for stocks that are hard to value and hard to arbitrage -- small, young,
  unprofitable, non-dividend-paying, extreme-growth and distressed firms -- and the
  reverse when sentiment is low. The aggregate-market predictability is far weaker
  than the cross-sectional conditioning, which is the heart of our null result on
  the plain S&P 500 index.

- **Baker, M. & Wurgler, J. (2007).** *Investor Sentiment in the Stock Market.*
  Journal of Economic Perspectives, 21(2), 129--151.
  The accessible companion. Lays out the "top-down" sentiment approach and the
  conditional, cross-sectional nature of the effect. Emphasizes that sentiment
  forecasts the *relative* returns of speculative vs safe stocks, not the level of
  the market index -- exactly why an index-level timing rule fails.

## Mechanisms, extensions, and challenges

- **Baker, M., Wurgler, J. & Yuan, Y. (2012).** *Global, Local, and Contagious
  Investor Sentiment.* Journal of Financial Economics, 104(2), 272--287.
  Extends the index globally; sentiment is partly contagious across borders and
  predicts the cross-section of returns within several major markets.

- **Stambaugh, R. F., Yu, J. & Yuan, Y. (2012).** *The Short of It: Investor
  Sentiment and Anomalies.* Journal of Financial Economics, 104(2), 288--302.
  Shows that many anomalies are stronger following high sentiment and are driven by
  the overpriced short leg -- a refinement of the BW story that locates the
  predictability in shorting frictions, not a long-only index bet.

- **Huang, D., Jiang, F., Tu, J. & Zhou, G. (2015).** *Investor Sentiment Aligned:
  A Powerful Predictor of Stock Returns.* Review of Financial Studies, 28(3),
  791--837.
  Builds a partial-least-squares "aligned" sentiment index and reports stronger
  aggregate-market predictability than the raw BW index -- a reminder that how the
  index is constructed matters, and that the raw BW level (what we test) is a weak
  market-timing signal.

## Post-publication decay

- **McLean, R. D. & Pontiff, J. (2016).** *Does Academic Research Destroy Stock
  Return Predictability?* Journal of Finance, 71(1), 5--32.
  Anomaly returns decay ~58% post-publication on average. Our sub-period split
  (strong contrarian tilt in 1965-1989, sign flip post-2008) is consistent with
  arbitrage eroding the index-level relationship after BW (2006) was published.

## Related desk studies

- **[Study 158 -- Super-Bowl](../../158-super-bowl/)** and other Folklore-family
  market-timing teardowns: the same base-rate / equity-risk-premium trap -- any rule
  that flattens you during up-months pays for it.
- **[Study 223 -- Same-Month-Seasonality](../../223-same-month-seasonality/)**:
  another conditioning variable (calendar month) tested with the same HAC machinery.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
  Matrix* (Econometrica).
- **Block-bootstrap Sharpe CI.** Politis & Romano (1994), *The Stationary
  Bootstrap* (JASA).
- **Sentiment index data.** Jeffrey Wurgler maintains the monthly index on the NYU
  Stern website (people.stern.nyu.edu/jwurgler); the series here is a hardcoded
  reconstruction of its documented regime structure for fully-offline reproduction.
