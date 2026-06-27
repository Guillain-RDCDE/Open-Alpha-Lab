# References & literature map -- Study 514 (Pastor-Stambaugh Liquidity Risk)

## The primary claim under test

- **Pastor, L. & Stambaugh, R. F. (2003).** "Liquidity Risk and Expected Stock Returns."
  *Journal of Political Economy*, 111(3), 642--685. The founding paper. They construct an
  aggregate market-liquidity series from the price-reversal signed-volume relation, define
  each stock's *liquidity beta* (its return sensitivity to innovations in aggregate
  liquidity), and show that stocks with higher liquidity betas earn higher average returns:
  a stock in the top decile of predicted liquidity beta out-earns the bottom decile by
  ~7.5%/yr (1966--1999). The premium is a *risk loading*, not an illiquidity *level*.

## Why this is distinct from the Amihud level (Study 140)

- **Amihud, Y. (2002).** "Illiquidity and Stock Returns: Cross-Section and Time-Series
  Effects." *Journal of Financial Markets*, 5(1), 31--56. Defines the |return|/dollar-volume
  illiquidity *measure*: high-illiquidity stocks earn a return *premium* for their level of
  illiquidity. Our [Study 140 -- Amihud-Illiquidity](../../140-amihud-illiquidity/) tests
  that LEVEL effect. Study 514 instead uses an Amihud cross-sectional *average* only as the
  ingredient for the *aggregate* liquidity series, and tests the *risk loading* (gamma) on
  innovations in that series -- a different, orthogonal channel.

## Theoretical backbone -- liquidity as a priced risk factor

- **Acharya, V. V. & Pedersen, L. H. (2005).** "Asset Pricing with Liquidity Risk."
  *Journal of Financial Economics*, 77(2), 375--410. A liquidity-adjusted CAPM with three
  liquidity betas (commonality in liquidity, return sensitivity to market liquidity, and
  liquidity sensitivity to market returns). Generalises Pastor-Stambaugh into a coherent
  equilibrium framework where liquidity risk carries a positive price.
- **Holmstrom, B. & Tirole, J. (2001).** "LAPM: A Liquidity-Based Asset Pricing Model."
  *Journal of Finance*, 56(5), 1837--1867. A macro-finance foundation for why aggregate
  liquidity shocks should be a priced state variable.

## Subsequent evidence, replication, and attenuation

- **Korajczyk, R. A. & Sadka, R. (2008).** "Pricing the Commonality Across Alternative
  Measures of Liquidity." *Journal of Financial Economics*, 87(1), 45--72. Extracts a
  common liquidity factor across several measures (Amihud, Pastor-Stambaugh, effective
  spread) and confirms it is priced -- but the magnitude depends heavily on the measure.
- **Ben-Rephael, A., Kadan, O., & Wohl, A. (2015).** "The Diminishing Liquidity Premium."
  *Journal of Financial and Quantitative Analysis*, 50(1-2), 197--229. Documents that the
  liquidity premium (both level and risk) has shrunk substantially since the 1960s as
  trading costs fell and markets deepened -- directly relevant to our 2010+ large-cap
  sample landing below the bar.
- **Li, H., Novy-Marx, R., & Velikov, M. (2019).** "Liquidity Risk and Asset Pricing."
  *Critical Finance Review*, 8(1-2), 223--255. A skeptical replication: the Pastor-Stambaugh
  liquidity-risk premium is sensitive to the construction of the aggregate series and is
  much weaker (often insignificant) on large-cap subsamples and post-2000 -- consistent with
  our null result.
- **McLean, R. D. & Pontiff, J. (2016).** "Does Academic Research Destroy Stock Return
  Predictability?" *Journal of Finance*, 71(1), 5--32. ~32% average post-publication
  attenuation; the liquidity-risk factor (published 2003) is well within the decay window.

## Survivorship and universe construction

- **Shumway, T. (1997).** "The Delisting Bias in CRSP Data." *Journal of Finance*, 52(1),
  327--340. Delistings are correlated with poor performance and with illiquidity. Removing
  delisted names -- the natural carriers of liquidity risk -- biases liquidity-factor returns
  and is the central caveat for our survivor basket.

## Method lineage (the desk's shared engine)

- **Newey, W. K. & West, K. D. (1987).** "A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica*,
  55(3), 703--708. The HAC long-run variance behind ``strategy.hac_tstat``.

## Related desk studies

- **[Study 140 -- Amihud-Illiquidity](../../140-amihud-illiquidity/)**: the illiquidity
  *level* premium -- the level cousin this study is deliberately distinct from.
- **[Study 238 -- Betting-Against-Beta](../../238-betting-against-beta/)**: another priced
  risk-loading sort (beta instead of liquidity beta), same rolling-loading infrastructure.
- **[Study 330 -- Low-Volatility-Anomaly](../../330-low-volatility-anomaly/)**: a related
  risk-sorting anomaly with HAC inference, placebo nulls, and explicit cost accounting.
