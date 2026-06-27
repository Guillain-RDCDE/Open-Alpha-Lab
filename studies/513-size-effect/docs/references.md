# References & literature map -- Study 513 (Size-Effect / Banz 1981 SMB)

## The primary claim under test

- **Banz, R. W. (1981).** "The Relationship Between Return and Market Value of Common Stocks."
  *Journal of Financial Economics*, 9(1), 3--18. The founding paper of the size effect: on
  NYSE stocks 1936--1975, small-cap firms earned higher risk-adjusted returns than large-cap
  firms, a relationship the CAPM could not explain. The original cross-sectional anomaly that
  this study replicates.
- **Reinganum, M. R. (1981).** "Misspecification of Capital Asset Pricing: Empirical Anomalies
  Based on Earnings' Yields and Market Values." *Journal of Financial Economics*, 9(1),
  19--46. Independent contemporaneous confirmation of the size effect and its interaction with
  the earnings-yield (E/P) anomaly.

## The size factor formalised

- **Fama, E. F. & French, K. R. (1992).** "The Cross-Section of Expected Stock Returns."
  *Journal of Finance*, 47(2), 427--465. Size (market cap) and book-to-market jointly capture
  the cross-section of average returns; beta adds little once size is included.
- **Fama, E. F. & French, K. R. (1993).** "Common Risk Factors in the Returns on Stocks and
  Bonds." *Journal of Financial Economics*, 33(1), 3--56. Introduces **SMB** (Small Minus Big)
  as one of the three factors -- the portfolio this study reconstructs in miniature.

## The January concentration

- **Keim, D. B. (1983).** "Size-Related Anomalies and Stock Return Seasonality: Further
  Empirical Evidence." *Journal of Financial Economics*, 12(1), 13--32. Documents that roughly
  half the annual size premium accrues in **January**, and much of that in the first few
  trading days -- the seasonality this study tests on the survivor basket (and finds reversed).
- **Roll, R. (1983).** "Vas Ist Das? The Turn-of-the-Year Effect and the Return Premia of
  Small Firms." *Journal of Portfolio Management*, 9(2), 18--28. The turn-of-the-year / tax-loss
  mechanism behind the January small-cap pop.

## Decay, fragility, and re-examination

- **Schwert, G. W. (2003).** "Anomalies and Market Efficiency." *Handbook of the Economics of
  Finance*, ch. 15. Shows the size effect **largely disappeared** after Banz's publication
  (1981) -- the post-publication decay this study's early-vs-late slice probes.
- **Horowitz, J. L., Loughran, T., & Savin, N. E. (2000).** "Three Analyses of the Firm Size
  Premium." *Journal of Empirical Finance*, 7(2), 143--153. Finds **no** reliable size premium
  in 1980--1996 US data -- the effect is not robust out-of-sample.
- **Asness, C., Frazzini, A., Israel, R., Moskowitz, T., & Pedersen, L. H. (2018).** "Size
  Matters, If You Control Your Junk." *Journal of Financial Economics*, 129(3), 479--509.
  Argues the size premium re-emerges *only after* controlling for quality (junk small-caps
  drag it down) -- a key caveat for any naive small-minus-large book.
- **van Dijk, M. A. (2011).** "Is Size Dead? A Review of the Size Effect in Equity Returns."
  *Journal of Banking & Finance*, 35(12), 3263--3274. A survey concluding the raw size effect
  is weak/unreliable in modern samples.

## Survivorship and replication caveats

- **Shumway, T. (1997).** "The Delisting Bias in CRSP Data." *Journal of Finance*, 52(1),
  327--340. Delistings correlate with poor performance; removing failed firms biases
  small-cap returns **upward** -- precisely the survivorship caveat this study names.
- **Hou, K., Xue, C., & Zhang, L. (2020).** "Replicating Anomalies." *Review of Financial
  Studies*, 33(5), 2019--2133. The size effect is among the weaker anomalies in their
  large-scale replication, often insignificant once micro-caps are excluded or NYSE breakpoints
  used.
- **McLean, R. D. & Pontiff, J. (2016).** "Does Academic Research Destroy Stock Return
  Predictability?" *Journal of Finance*, 71(1), 5--32. ~58% post-publication decay on average;
  the size effect, published 1981, is a textbook example of attenuation.

## Method lineage (the desk's shared engine)

- **Newey, W. K. & West, K. D. (1987).** "A Simple, Positive Semi-Definite, Heteroskedasticity
  and Autocorrelation Consistent Covariance Matrix." *Econometrica*, 55(3), 703--708. The HAC
  long-run variance estimator behind `strategy.hac_tstat`.

## Related desk studies

- **[Study 238 -- Betting-Against-Beta](../../238-betting-against-beta/)**: the sibling
  cross-sectional sort (rank by rolling beta), same rolling-sort / equal-weight / HAC infra.
- **[Study 330 -- Low-Volatility-Anomaly](../../330-low-volatility-anomaly/)**: the calm-vs-wild
  ETF race -- a different risk-based cross-sectional anomaly on the same desk bench.
- **[Study 122 -- Gross-Profitability](../../122-gross-profitability/)**: a fundamental factor
  sort (Novy-Marx quality) -- the quality control Asness et al. (2018) say resurrects size.
