# References & literature map -- Study 508 (Momentum-Crashes)

## The primary claim under test

- **Daniel, K. & Moskowitz, T. J. (2016).** "Momentum Crashes." *Journal of Financial
  Economics*, 122(2), 221--247. The paper this study replicates. Cross-sectional momentum
  earns a strong unconditional premium but suffers rare, severe **crashes** that cluster in
  *panic states* -- the rebound out of a bear market, when the past-loser (short) leg snaps back
  violently. Momentum's conditional mean and beta are **regime-dependent** (an option-like,
  written-call payoff in bear-market rebounds). Their fix: a **dynamic / vol-scaled** momentum
  that targets constant volatility lifts the Sharpe and tames the left tail. Our real-tape run
  reproduces the *shape* result (loser-leg-driven crashes, negative skew halved by vol-scaling)
  on a small survivor basket while the *level* premium is flat.

## The momentum factor itself

- **Jegadeesh, N. & Titman, S. (1993).** "Returns to Buying Winners and Selling Losers:
  Implications for Stock Market Efficiency." *Journal of Finance*, 48(1), 65--91. The founding
  cross-sectional momentum paper -- buy the trailing 3-12 month winners, short the losers. The
  12-1 (skip the most recent month, to dodge short-term reversal) is the canonical specification
  we build.
- **Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013).** "Value and Momentum
  Everywhere." *Journal of Finance*, 68(3), 929--985. Momentum is pervasive across markets and
  asset classes and negatively correlated with value -- context for why the factor is so widely
  traded and so crash-exposed.
- **Barroso, P. & Santa-Clara, P. (2015).** "Momentum Has Its Moments." *Journal of Financial
  Economics*, 116(1), 111--120. The parallel vol-scaling result: managing momentum's own
  realised volatility roughly doubles its Sharpe and removes the crash risk -- the same "risk-
  managed momentum" we implement as the repair leg.

## The crash mechanism and regime dependence

- **Grundy, B. D. & Martin, J. S. (2001).** "Understanding the Nature of the Risks and the
  Source of the Rewards to Momentum Investing." *Review of Financial Studies*, 14(1), 29--78.
  Momentum's time-varying market beta -- the past-winner/loser legs load oppositely on the
  market depending on the prior market run -- foreshadows the regime-dependence Daniel-Moskowitz
  formalise.
- **Cooper, M. J., Gutierrez, R. C., & Hameed, A. (2004).** "Market States and Momentum."
  *Journal of Finance*, 59(3), 1345--1365. Momentum profits depend on the *market state*:
  strongly positive following up-markets, negative or zero following down-markets -- the
  empirical backbone of our bear/calm regime split.
- **Daniel, K., Jagannathan, R., & Kim, S. (2019).** "A Hidden Markov Model of Momentum."
  Working paper / NBER. Models the latent "turbulent" state in which momentum crashes occur and
  shows a state-aware overlay improves performance -- a more sophisticated cousin of our
  bear-indicator conditioning.

## Costs, decay, and implementability

- **Novy-Marx, R. & Velikov, M. (2016).** "A Taxonomy of Anomalies and Their Trading Costs."
  *Review of Financial Studies*, 29(1), 104--147. Momentum has high turnover; net of realistic
  trading costs much of the gross premium erodes -- consistent with our gross-to-net gap.
- **McLean, R. D. & Pontiff, J. (2016).** "Does Academic Research Destroy Stock Return
  Predictability?" *Journal of Finance*, 71(1), 5--32. ~32% post-publication attenuation;
  momentum's large-cap premium has decayed materially in the post-2009 sample we test.

## Survivorship bias and universe construction

- **Shumway, T. (1997).** "The Delisting Bias in CRSP Data." *Journal of Finance*, 52(1),
  327--340. Delistings correlate with poor performance; removing them biases factor returns
  upward. For momentum this is acute -- the past-loser short leg's worst names are exactly the
  firms that trended into delisting, which our survivor basket cannot include.

## Method lineage (the desk's shared engine)

- **Newey, W. K. & West, K. D. (1987).** "A Simple, Positive Semi-Definite, Heteroskedasticity
  and Autocorrelation Consistent Covariance Matrix." *Econometrica*, 55(3), 703--708. The HAC
  long-run variance estimator in [`strategy.hac_tstat`](../momentum_crashes/strategy.py).

## Related desk studies

- **[Study 507 -- Cross-Sectional-Momentum](../507-cross-sectional-momentum/)**: the same 12-1
  WML book, asking whether the *level* premium survives (it does not). 508 reuses that basket and
  asks the *crash* question instead.
- **[Study 24 -- Stampede](../24-stampede/)**: cross-sectional momentum on the full S&P 500 with
  a first look at the crash cost.
- **[Study 25 -- Clean-Slate](../25-clean-slate/)**: residual momentum -- the "crash-dodging"
  cousin that strips out the time-varying market beta the crash rides on.
- **[Study 505 -- Left-Tail-Momentum](../505-left-tail-momentum/)**: sorting on downside-tail
  risk rather than the spread's own tail.
- **[Study 237 -- Residual-Momentum](../237-residual-momentum/)**: does residual-return momentum
  dodge the crashes (a direct test of the Daniel-Moskowitz crash-avoidance angle)?
