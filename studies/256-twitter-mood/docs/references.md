# References & literature map -- Study 256 (Twitter-Mood)

## The canonical claim

- **Bollen, J., Mao, H. & Zeng, X. (2011).** *Twitter mood predicts the stock
  market.* Journal of Computational Science, 2(1), 1-8.
  The founding paper. Two mood-tracking tools (OpinionFinder for positive/negative
  and GPOMS for six dimensions: Calm, Alert, Sure, Vital, Kind, Happy) are run on
  ~9.7M tweets from Feb-Dec 2008. The "Calm" dimension is reported to
  Granger-cause the Dow Jones 3-4 days ahead, lifting a Self-Organizing Fuzzy
  Neural Network directional model to **86.7% next-day accuracy**. This is the
  result the whole "social-media alpha" industry was built on.

## Failed replications and methodological critiques

- **Lachanski, M. & Pav, S. (2017).** *Shy of the Character Limit: "Twitter Mood
  Predicts the Stock Market" Revisited.* Econ Journal Watch, 14(3), 302-345.
  The definitive teardown: the Bollen result does not replicate. The Granger
  causality is fragile to lag choice and sample window, the n is tiny (~10
  trading months), and the headline 87% accuracy figure comes from an in-sample
  model fit. Out of sample the edge vanishes.

- **Brown, E. (2012).** *Will Twitter Make You a Better Investor? A Look at Sentiment,
  User Reputation and Their Effect on the Stock Market.* SAIS Proceedings.
  Finds the predictive content of aggregate Twitter sentiment for daily returns
  is weak to nonexistent once proper out-of-sample testing is applied.

- **Multiple-comparisons / lag-mining trap.** Sweeping Granger lags (1..k days)
  and reporting the most significant inflates the false-positive rate. With
  k=5 lags the Bonferroni 5% bar is |t| ~ 2.58 -- our real-tape sweep peaks at
  |t| = 0.90 and clears nothing.

## Mechanisms, base rates, and the tiny-n problem

- **Tetlock, P. C. (2007).** *Giving Content to Investor Sentiment: The Role of
  Media in the Stock Market.* Journal of Finance, 62(3), 1139-1168.
  The more careful media-sentiment baseline: pessimism in news text predicts
  short-horizon downward pressure that *reverts* -- a transient liquidity/price-
  pressure effect, not a persistent forecastable alpha. A useful contrast to the
  Bollen overclaim.

- **The base-rate trap.** The S&P drifts up ~53% of trading days; any directional
  rule that is long-ish most of the time inherits that hit-rate for free. The
  honest baseline is the unconditional up-rate, not a 50% coin -- our directional
  hit-rate (51.3%) sits right on its 50.8% baseline.

- **Window contamination.** Bollen's window (2008) is dominated by the financial
  crisis. Any rule that ends up net-short during the crash looks brilliant for a
  reason unrelated to mood -- a confound our long-short variant makes explicit.

## Method lineage

- **Granger, C. W. J. (1969).** *Investigating Causal Relations by Econometric
  Models and Cross-spectral Methods.* Econometrica, 37(3), 424-438. The lead-lag
  "causality" test Bollen relied on.
- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
  Matrix* (Econometrica). Used for all slope and return t-stats here.
- **Permutation inference.** Time-shuffling the predictor to build a null that
  respects the return marginal -- the cleanest "could this be zero?" test for a
  short, autocorrelated daily series.

## Related desk studies

- **[Study 158 -- Super-Bowl](../../158-super-bowl/)**: another folklore predictor
  killed by the base-rate trap and a tiny n.
- **[Study 223 -- Same-Month-Seasonality](../../223-same-month-seasonality/)**: a
  *real* cross-sectional seasonal -- the contrast with this mirage is instructive.
