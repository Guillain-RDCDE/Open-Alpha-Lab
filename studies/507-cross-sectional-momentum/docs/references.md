# References & literature map -- Study 507 (Cross-Sectional-Momentum)

## The primary claim under test

- **Jegadeesh, N. & Titman, S. (1993).** "Returns to Buying Winners and Selling Losers:
  Implications for Stock Market Efficiency." *Journal of Finance*, 48(1), 65--91. The founding
  paper for cross-sectional equity momentum. Ranking US stocks by trailing 3--12 month return
  and buying winners / shorting losers earned ~1%/month over 1965--1989. The 12-1 variant
  (12-month lookback, skip the most recent month to dodge short-term reversal) is the canonical
  relative-strength factor we replicate.
- **Jegadeesh, N. & Titman, S. (2001).** "Profitability of Momentum Strategies: An Evaluation
  of Alternative Explanations." *Journal of Finance*, 56(2), 699--720. An out-of-sample
  confirmation on 1990--1998 that momentum persisted after publication of the original paper.

## Why the effect should exist -- and why it may have decayed

- **Carhart, M. M. (1997).** "On Persistence in Mutual Fund Performance." *Journal of Finance*,
  52(1), 57--82. Adds momentum (UMD/WML) as the fourth factor to the Fama-French model -- the
  factor's canonical asset-pricing home.
- **Asness, C. S., Moskowitz, T. J. & Pedersen, L. H. (2013).** "Value and Momentum Everywhere."
  *Journal of Finance*, 68(3), 929--985. Momentum is pervasive across asset classes and
  countries and negatively correlated with value -- the steelman for momentum as a real,
  diversifying premium.
- **Barroso, P. & Santa-Clara, P. (2015).** "Momentum Has Its Moments." *Journal of Financial
  Economics*, 116(1), 111--120. Documents the catastrophic, forecastable momentum-crash left
  tail and shows volatility-scaling roughly doubles the strategy's Sharpe -- relevant to the
  -53% drawdown we observe.
- **Daniel, K. & Moskowitz, T. J. (2016).** "Momentum Crashes." *Journal of Financial
  Economics*, 122(2), 221--247. The loser leg violently rebounds in panic-to-rebound regimes,
  producing the deep, infrequent crashes that dominate momentum's tail risk.

## Decay, costs, and the survivor caveat

- **McLean, R. D. & Pontiff, J. (2016).** "Does Academic Research Destroy Stock Return
  Predictability?" *Journal of Finance*, 71(1), 5--32. Documents ~32% post-publication
  attenuation of anomalies; momentum, heavily traded since 1993, is a prime candidate for decay
  in modern large-cap samples -- consistent with the flat result here.
- **Korajczyk, R. A. & Sadka, R. (2004).** "Are Momentum Profits Robust to Trading Costs?"
  *Journal of Finance*, 59(3), 1039--1082. Momentum's high turnover makes net profits sensitive
  to transaction costs and capacity -- the gross-to-net erosion this study charges.
- **Novy-Marx, R. & Velikov, M. (2016).** "A Taxonomy of Anomalies and Their Trading Costs."
  *Review of Financial Studies*, 29(1), 104--147. Momentum is among the highest-turnover
  anomalies; net of realistic costs the premium shrinks materially.
- **Shumway, T. (1997).** "The Delisting Bias in CRSP Data." *Journal of Finance*, 52(1),
  327--340. Delistings correlate with poor performance; removing failed firms biases factor
  returns upward. For momentum the bias is acute -- the deleted names are the persistent losers
  the short leg would harvest, so a survivor basket understates the loser-leg's drag and
  inflates any apparent WML premium. This is why we name survivorship on the SIGNAL axis.

## Method lineage (the desk's shared engine)

- **Newey, W. K. & West, K. D. (1987).** "A Simple, Positive Semi-Definite, Heteroskedasticity
  and Autocorrelation Consistent Covariance Matrix." *Econometrica*, 55(3), 703--708. The HAC
  long-run variance estimator behind `strategy.hac_tstat`.

## Related desk studies

- **[Study 24 -- Stampede](../../24-stampede/)**: the same Jegadeesh-Titman 12-1 WML on the
  *full* modern S&P 500 cross-section (Weak/Fragile, Severe crash). Study 507 is the
  *retail-realistic small-basket* cousin with the decile-vs-quintile question and survivorship
  named on the signal axis.
- **[Study 25 -- Clean-Slate](../../25-clean-slate/)**: residual momentum -- the "cleaner cousin"
  that strips the market factor before ranking.
- **[Study 237 -- Residual-Momentum](../../237-residual-momentum/)**: does residual-return
  momentum dodge the crashes?
- **[Study 238 -- Betting-Against-Beta](../../238-betting-against-beta/)**: a sibling
  cross-sectional sort on the same large-cap survivor infrastructure (rank, leg, HAC inference).
- **[Study 50 -- High-Water](../../50-high-water/)** and
  **[Study 236 -- Fifty-Two-Week-High](../../236-fifty-two-week-high/)**: the 52-week-high
  effect -- momentum wearing a different hat.
