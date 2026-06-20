# References & literature map -- Study 325 (Crypto-Fear-Greed)

## The claim and its source

- **alternative.me, *Crypto Fear & Greed Index*** (live since Feb-2018).
  https://alternative.me/crypto/fear-and-greed-index/ -- the 0--100 composite
  gauge whose contrarian folk rule ("buy Extreme Fear, sell Extreme Greed") this
  study tests. The published index blends volatility (25%), market
  momentum/volume (25%), social media (15%), surveys (15%), Bitcoin dominance
  (10%) and Google Trends (10%) -- the first two, which we proxy from price, are
  the largest weights.
- **Warren Buffett**, Berkshire Hathaway shareholder letters -- the origin of
  "be fearful when others are greedy, and greedy when others are fearful," the
  intuition the gauge sells.

## Sentiment, attention and crypto returns

- Baker, M. & Wurgler, J. (2006), *Investor Sentiment and the Cross-Section of
  Stock Returns*, Journal of Finance 61(4) -- the canonical equity-sentiment
  framework; aggregate sentiment predicts cross-sectional, not near-term index,
  returns.
- Brown, G. & Cliff, M. (2004), *Investor Sentiment and the Near-Term Stock
  Market*, Journal of Empirical Finance 11(1) -- sentiment has little near-term
  index-timing power, the equity analogue of this study's null.
- Da, Z., Engelberg, J. & Gao, P. (2015), *The Sum of All FEARS: Investor
  Sentiment and Asset Prices*, Review of Financial Studies 28(1) -- a
  search-volume "FEARS" index; high attention/fear is followed by short reversal
  then reversion, the mechanism a contrarian rule hopes for.
- Liu, Y. & Tsyvinski, A. (2021), *Risks and Returns of Cryptocurrency*, Review
  of Financial Studies 34(6) -- crypto returns are dominated by a strong
  **momentum** factor and attention proxies, not contrarian mean reversion.
- Liu, Y., Tsyvinski, A. & Wu, X. (2022), *Common Risk Factors in
  Cryptocurrency*, Journal of Finance 77(2) -- a crypto size/momentum factor
  model; momentum is the dominant tradable signal, working *against* a
  fear-buying overlay.

## Method (the desk's shared apparatus)

- Newey, W. & West, K. (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*,
  Econometrica 55(3) -- the HAC *t*-stat that decides the Signal axis.
- Politis, D. & Romano, J. (1994), *The Stationary Bootstrap*, JASA 89(428) --
  block-bootstrap resampling that preserves volatility clustering (i.i.d.
  resampling would destroy it), behind the Sharpe CI.
- Lo, A. (2002), *The Statistics of Sharpe Ratios*, Financial Analysts Journal
  58(4) -- autocorrelation-aware Sharpe inference.
- White, H. (2000), *A Reality Check for Data Snooping*, Econometrica 68(5) --
  the data-snooping correction the desk applies when a cross-section or many
  thresholds are searched.

## Sibling studies on this bench

- **255-fear-greed-index** -- the *equities* CNN Fear & Greed contrarian test
  (curated sentiment table on ^GSPC weekly). This study (325) is the **crypto /
  BTC** counterpart with a *price-derived* gauge, deliberately distinct: a
  different asset, a different gauge construction, a daily clock.
- **210-crypto-trend**, **251-crypto-reversal** -- crypto trend/reversal
  teardowns that establish the momentum regime which sinks the contrarian rule
  here.
