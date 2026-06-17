# References & literature map -- Study 257 (AAII-Sentiment)

## The survey itself

- **American Association of Individual Investors (1987-present).** *AAII Investor
  Sentiment Survey.* A weekly poll (since 24 July 1987) of AAII members asking
  whether they are bullish, neutral, or bearish on the stock market over the next
  six months. The headline gauge is the **bull-bear spread** (`%bull - %bear`);
  the long-run averages are roughly 37.5% bull / 31.5% neutral / 31.0% bear.
  AAII publishes the full weekly history and explicitly frames extreme readings
  as *contrarian*. This study uses a curated **monthly** snapshot, not the raw
  redistributed weekly tape.

## Does individual-investor sentiment predict returns?

- **Brown, G. W. & Cliff, M. T. (2004).** *Investor Sentiment and the Near-Term
  Stock Market.* Journal of Empirical Finance, 11(1), 1-27. Survey sentiment
  (including AAII) is strongly correlated with *contemporaneous* returns but has
  little reliable power to forecast *near-term* returns -- sentiment looks more
  like a coincident than a leading indicator.

- **Baker, M. & Wurgler, J. (2006).** *Investor Sentiment and the Cross-Section
  of Stock Returns.* Journal of Finance, 61(4), 1645-1680. Builds a composite
  sentiment index and shows sentiment predicts the *cross-section* (small,
  young, volatile stocks) more than the aggregate market -- the headline index
  timing effect is weak, consistent with our market-level Weak verdict.

- **Fisher, K. L. & Statman, M. (2000).** *Investor Sentiment and Stock Returns.*
  Financial Analysts Journal, 56(2), 16-23. Finds a *negative* (contrarian)
  relation between individual-investor (AAII) sentiment and subsequent S&P 500
  returns -- the direction we recover -- but the magnitude is modest and noisy.

- **Solt, M. E. & Statman, M. (1988).** *How Useful Is the Sentiment Index?*
  Financial Analysts Journal, 44(5), 45-55. An early, skeptical look at sentiment
  indices as market-timing tools; concludes they add little once the market's
  upward drift is accounted for.

## Why a directional pattern can still fail as a strategy

- **Smith, P. (1987-2024) / AAII commentary.** Note that the contrarian edge is
  concentrated at *extremes* that recur only a handful of times a generation
  (1987, 2002-03, 2009, 2020, 2022); between extremes the survey is pure noise.

- **McLean, R. D. & Pontiff, J. (2016).** *Does Academic Research Destroy Stock
  Return Predictability?* Journal of Finance, 71(1), 5-32. Anomaly returns decay
  ~58% post-publication; a widely-watched public survey is the archetypal
  arbitraged signal.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
  Matrix* (Econometrica) -- used for the regime, long-short, and regression t-stats.

## Related desk studies

- **[Study 158 -- Super-Bowl](../../158-super-bowl/)** and the broader folklore
  family: binary "indicators" that ride the market's unconditional up-drift.
- **[Study 223 -- Same-Month-Seasonality](../../223-same-month-seasonality/)**:
  the synthetic-panel + cached-real-series pattern this study mirrors.
- VIX-term / MOVE / put-call studies in the desk: other "fear gauges" pitched as
  contrarian timing tools.
