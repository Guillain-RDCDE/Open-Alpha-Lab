# References & literature map -- Study 237 (Residual-Momentum)

## The canonical claim

- **Blitz, D., Huij, J. & Martens, M. (2011).** *Residual Momentum.*
  Journal of Empirical Finance, 18(3), 506--521.
  The founding paper: sort stocks by their trailing 12-month *CAPM residual*
  (idiosyncratic) return rather than raw price return. The authors show that residual
  momentum (a) delivers similar gross returns to raw momentum on NYSE/AMEX stocks
  1926-2009, (b) has substantially lower systematic (beta) risk, and (c) suffers
  smaller drawdowns in "momentum crash" episodes (particularly January and market
  reversals). The key insight: standard momentum bundles systematic and idiosyncratic
  components; stripping market beta isolates the true behavioural signal.

- **Jegadeesh, N. & Titman, S. (1993).** *Returns to Buying Winners and Selling Losers:
  Implications for Stock Market Efficiency.* Journal of Finance, 48(1), 65--91.
  The reference paper for raw momentum (12-1 months). Residual momentum is a
  proposed refinement of this strategy to reduce crash risk.

## The mechanism: why residual returns might predict future returns

- **Daniel, K. & Moskowitz, T. J. (2016).** *Momentum Crashes.*
  Journal of Financial Economics, 122(2), 221--247.
  Documents that standard momentum strategies experience rare but severe crashes,
  especially in down-market recoveries (e.g., 2009). These crashes arise because
  the short leg (past losers) has high market beta -- it rallies sharply when the
  market recovers. Residual momentum aims to reduce this exposure by sorting on
  the idiosyncratic (beta-stripped) component.

- **Grundy, B. D. & Martin, J. S. (2001).** *Understanding the Nature of the Risks
  and the Source of the Rewards to Momentum Investing.* Review of Financial Studies,
  14(1), 29--78.
  Shows that raw momentum portfolios carry time-varying systematic risk (beta changes
  with the sign of the market). Sorting on residuals partially corrects for this by
  removing the contemporaneous beta from the signal.

## Post-publication evidence and limits

- **Fuertes, A.-M., Miffre, J. & Tan, W.-H. (2009).** *Momentum Profits, Non-normality
  Risks and the Business Cycle.* Applied Financial Economics, 19(12), 935--953.
  Finds that idiosyncratic momentum (similar to residual momentum) is more robust
  across business cycle phases than raw momentum.

- **Novy-Marx, R. (2012).** *Is Momentum Really Momentum?*
  Journal of Financial Economics, 103(3), 429--453.
  Challenges standard momentum explanations; shows intermediate-horizon (7-12 month)
  returns drive most of the momentum effect. Residual momentum uses a similar
  intermediate-horizon window.

- **McLean, R. D. & Pontiff, J. (2016).** *Does Academic Research Destroy Stock Return
  Predictability?* Journal of Finance, 71(1), 5--32.
  Documents ~58% post-publication decay in anomaly returns. The Blitz et al. (2011)
  residual momentum paper was published in 2011, and our sub-period analysis
  shows 2016-2026 turns negative (-0.9%/yr), consistent with post-publication decay.

## The survivorship-bias dimension

- **Shumway, T. (1997).** *The Delisting Bias in CRSP Data.* Journal of Finance,
  52(1), 327--340.
  Delisted stocks (failed, acquired) earn extreme negative returns but are often
  excluded from survivorship-biased panels. Our universe (current large-cap basket
  backwards) is survivorship-biased; all results are upper bounds.

## Related desk studies

- **[Study 24 -- Stampede](../../24-stampede/)**: raw 12-1 price momentum, the
  baseline that residual momentum claims to improve upon. The same cross-sectional
  engine, different signal construction (raw log return vs CAPM residual).
- **[Study 196 -- Long-Term-Reversal](../../196-long-term-reversal/)**: the
  mean-reversion at long horizons (36-60 months), the opposite end of the
  return-autocorrelation spectrum from momentum.
- **[Study 33 -- Slingshot](../../33-slingshot/)**: short-term reversal (1-month),
  the contrarian at the other extreme.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
  Matrix* (Econometrica) -- [`strategy.summarize`](../residual_momentum/strategy.py).
- **Rolling OLS beta.** Standard CAPM regression in [`strategy.rolling_betas`](../residual_momentum/strategy.py).
- **Survivorship notation.** Shumway (1997) -- named on the Signal axis.
