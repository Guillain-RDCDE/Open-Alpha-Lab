# References & literature map -- Study 530 (Book-To-Market-Value)

## The primary claim under test

- **Fama, E. F. & French, K. R. (1992).** "The Cross-Section of Expected Stock Returns."
  *Journal of Finance*, 47(2), 427--465. The canonical paper. Sorting US stocks by
  book-to-market equity (B/M) produces a monotone return spread: high-B/M ("value") stocks
  earn higher average returns than low-B/M ("growth") stocks, and this spread is *not*
  explained by market beta. The B/M sort is the origin of the value factor.
- **Fama, E. F. & French, K. R. (1993).** "Common Risk Factors in the Returns on Stocks and
  Bonds." *Journal of Financial Economics*, 33(1), 3--56. Formalises the value premium as the
  **HML** ("High Minus Low" book-to-market) factor in the three-factor model -- long the
  high-B/M tercile, short the low-B/M tercile, the exact long-value / short-growth hedge this
  study replicates.

## Why the premium should exist -- competing explanations

- **Lakonishok, J., Shleifer, A., & Vishny, R. W. (1994).** "Contrarian Investment,
  Extrapolation, and Risk." *Journal of Finance*, 49(5), 1541--1578. The behavioural account:
  investors over-extrapolate past growth, overpricing glamour (low-B/M) stocks and underpricing
  value (high-B/M) stocks; the premium is a correction of that mistake -- not compensation for risk.
- **Fama, E. F. & French, K. R. (1995).** "Size and Book-to-Market Factors in Earnings and
  Returns." *Journal of Finance*, 50(1), 131--155. The risk account: high-B/M firms are
  distressed, with persistently low earnings; the value premium compensates for that
  distress/financial-risk exposure.
- **Zhang, L. (2005).** "The Value Premium." *Journal of Finance*, 60(1), 67--103. A
  production-based, real-options explanation: value firms carry more unproductive capital that
  is costly to scale down in bad times, making them riskier exactly when risk is most painful.

## The post-2007 disappearance of the value premium (decisive for this study's window)

- **Fama, E. F. & French, K. R. (2021).** "The Value Premium." *Review of Asset Pricing
  Studies*, 11(1), 105--121. The authors themselves document that the value premium is much
  weaker (and statistically fragile) in the post-1991 / post-publication period, especially
  in large-caps -- directly relevant to our 2022-2025 large-cap window.
- **Arnott, R., Harvey, C. R., Kalesnik, V., & Linnainmaa, J. (2021).** "Reports of Value's
  Death May Be Greatly Exaggerated." *Financial Analysts Journal*, 77(1), 44--67. Argues the
  2010s value drawdown was driven by *re-pricing* (growth getting more expensive) rather than
  a vanished premium -- but concedes the realised premium over the decade was deeply negative,
  consistent with the negative hedge we measure.
- **Israel, R., Laursen, K., & Richardson, S. (2021).** "Is (Systematic) Value Investing
  Dead?" *Journal of Portfolio Management*, 47(2). Documents the historic 2018-2020 value
  drawdown and the mega-cap-growth dominance that frames our sample.

## Measurement, survivorship, and trading costs

- **Shumway, T. (1997).** "The Delisting Bias in CRSP Data." *Journal of Finance*, 52(1),
  327--340. Delistings correlate with poor performance; removing failed firms biases factor
  returns upward. Deep-value (high-B/M) names are the most likely to delist into bankruptcy,
  so survivorship bias inflates the *long* leg of a B/M sort specifically.
- **Novy-Marx, R. & Velikov, M. (2016).** "A Taxonomy of Anomalies and Their Trading Costs."
  *Review of Financial Studies*, 29(1), 104--147. B/M-based value has *low* turnover relative
  to most anomalies (B/M ranks are sticky) -- consistent with the ~16%/yr turnover we measure;
  trading costs are not what kills value here.
- **Newey, W. K. & West, K. D. (1987).** "A Simple, Positive Semi-Definite, Heteroskedasticity
  and Autocorrelation Consistent Covariance Matrix." *Econometrica*, 55(3), 703--708. The HAC
  long-run variance estimator behind the *t*-stat in
  [`strategy.summary`](../book_to_market_value/strategy.py).

## Related desk studies (other value lenses -- this is the *pure* B/M factor)

- **[Study 124 -- Cash-Flow-Yield](../../124-cash-flow-yield/)**: operating-cash-flow yield --
  a cash-based value signal; this study isolates the *balance-sheet* B/M instead.
- **[Study 121 -- Magic-Formula](../../121-magic-formula/)**: Greenblatt's earnings-yield +
  ROIC combination -- value *plus* quality, not the raw value sort.
- **[Study 243 -- Graham-NCAV](../../243-graham-ncav/)**: net-current-asset-value deep value --
  a stricter liquidation-value screen than book equity.
- **[Study 238 -- Betting-Against-Beta](../../238-betting-against-beta/)**: the same annual
  cross-sectional sort + HAC inference + survivor-basket machinery, applied to risk rather
  than value.
