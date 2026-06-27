# References & literature map -- Study 518 (Time-Series-Momentum)

## The primary claim under test

- **Moskowitz, T. J., Ooi, Y. H. & Pedersen, L. H. (2012).** "Time Series Momentum." *Journal
  of Financial Economics*, 104(2), 228--250. The founding paper. An asset's OWN past 12-month
  excess return positively predicts its next-month return across 58 instruments in four asset
  classes (equity-index futures, bond futures, commodities, currencies). A diversified,
  vol-scaled long/short "trend everywhere" book delivered a high Sharpe with persistent crisis
  alpha. The own-sign, vol-scaled, cross-asset recipe we replicate.

## Why the effect should exist -- and why it may have decayed

- **Hurst, B., Ooi, Y. H. & Pedersen, L. H. (2017).** "A Century of Evidence on Trend-Following
  Investing." *Journal of Portfolio Management*, 44(1). Extends time-series momentum back to
  1880 across asset classes -- the steelman that trend is a durable, century-long premium, not a
  recent artefact. Also documents the post-2009 "trend drought" that thinned returns.
- **Baltas, N. & Kosowski, R. (2013).** "Momentum Strategies in Futures Markets and
  Trend-Following Funds." Working paper / SSRN. Confirms TSMOM in futures and links it to the
  returns of managed-futures (CTA) funds -- the industry that actually trades this signal.
- **Asness, C. S., Moskowitz, T. J. & Pedersen, L. H. (2013).** "Value and Momentum Everywhere."
  *Journal of Finance*, 68(3), 929--985. Momentum (cross-sectional cousin) is pervasive across
  asset classes and countries and negatively correlated with value -- context for why trend
  diversifies a portfolio.
- **Barroso, P. & Santa-Clara, P. (2015).** "Momentum Has Its Moments." *Journal of Financial
  Economics*, 116(1), 111--120. Volatility-scaling roughly doubles a momentum strategy's Sharpe
  by taming the crash -- directly relevant to our third-axis finding that inverse-vol sizing
  beats an equal-notional sign book.

## Distinct from -- and decay/cost caveats

- **Jegadeesh, N. & Titman, S. (1993).** "Returns to Buying Winners and Selling Losers."
  *Journal of Finance*, 48(1), 65--91. CROSS-sectional momentum -- rank assets *against each
  other*, long winners / short losers. Study 518 is the *time-series* cousin: each asset is
  traded against ITS OWN history, not its peers. Replicated separately in
  [Study 507 -- Cross-Sectional-Momentum](../507-cross-sectional-momentum/).
- **Faber, M. T. (2007).** "A Quantitative Approach to Tactical Asset Allocation." *Journal of
  Wealth Management*, 9(4), 69--79. Price-vs-10-month-SMA trend timing, *long-or-flat* on a
  single asset -- replicated in [Study 110 -- Faber-Timing](../110-faber-timing/). TSMOM differs:
  the signal is the *sign of the trailing return* (not price-vs-average), the book is *long AND
  short*, and positions are *vol-scaled and diversified across asset classes*.
- **Huang, D., Li, J., Wang, L. & Zhou, G. (2020).** "Time Series Momentum: Is It There?"
  *Journal of Financial Economics*, 135(3), 774--794. A direct challenge: argues much of the
  apparent TSMOM predictability is driven by a few assets and disappears under proper pooled
  tests -- consistent with the sub-2 *t* we find on a small, modern, survivor ETF basket.
- **McLean, R. D. & Pontiff, J. (2016).** "Does Academic Research Destroy Stock Return
  Predictability?" *Journal of Finance*, 71(1), 5--32. ~32% post-publication anomaly
  attenuation; TSMOM, heavily traded by CTAs since 2012, is a prime decay candidate -- consistent
  with the thin, regime-dependent premium here.

## Method lineage (the desk's shared engine)

- **Newey, W. K. & West, K. D. (1987).** "A Simple, Positive Semi-Definite, Heteroskedasticity
  and Autocorrelation Consistent Covariance Matrix." *Econometrica*, 55(3), 703--708. The HAC
  long-run variance estimator behind `strategy.hac_tstat`.

## Related desk studies

- **[Study 507 -- Cross-Sectional-Momentum](../507-cross-sectional-momentum/)**: the
  rank-against-peers cousin (Jegadeesh-Titman). 518 ranks each asset against its own history.
- **[Study 110 -- Faber-Timing](../110-faber-timing/)**: price-vs-SMA, long-or-flat single-asset
  trend. 518 is sign-of-return, long/short, vol-scaled, cross-asset.
- **[Study 508 -- Momentum-Crashes](../508-momentum-crashes/)** and
  **[Study 509 -- Intermediate-Momentum](../509-intermediate-momentum/)**: the crash tail and
  lookback-horizon questions of the momentum family.
- **[Study 144 -- Permanent-Portfolio](../144-permanent-portfolio/)** and
  **[Study 97 -- Balancing-Act](../97-balancing-act/)**: the cross-asset diversification backdrop
  TSMOM trades on top of.
