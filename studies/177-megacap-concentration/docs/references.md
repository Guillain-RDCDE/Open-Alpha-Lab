# References & literature map -- Study 177 (Megacap-Concentration)

## The claim under test

- **The folk thesis.** "The biggest companies in the S&P 500 stay big because they have
  the widest moats, the best capital allocation, and the most durable earnings growth.
  Just buy the top 7 (or 10) by market cap each year and hold them. You'll beat the
  index." Popular in finance social media since ~2021, amplified by Magnificent-Seven
  narratives. The modern version is a reaction to the cap-weight tilt in SPY toward the
  same names. We steelman it as: "the top-N S&P 500 names by (prior year-end) market cap
  systematically earn higher annual returns than the average member, across market regimes."

## The countervailing evidence: the size premium

- **Banz (1981).** "The Relationship Between Return and Market Value of Common Stocks."
  *Journal of Financial Economics*, 9(1), 3-18. The original size-effect paper: small-cap
  stocks earn higher average returns than large-cap stocks, with the result holding from
  1926-1975. The concentration claim requires this to be dead.
- **Fama & French (1992, 1993).** "The Cross-Section of Expected Stock Returns" (*Journal
  of Finance*) and "Common Risk Factors in the Returns on Stocks and Bonds" (*Journal of
  Financial Economics*). The three-factor model adds SMB (small-minus-big) to the market
  factor: large-cap *is* a negative loading on SMB, i.e., historically a *lower* expected
  return. Our pre-2015 result (-14 ppt/year for the top-10) is exactly this effect.
- **Fama & French (2012).** "Size, Value, and Momentum in International Stock Returns."
  *Journal of Financial Economics*, 105(3), 457-472. Size effects across 23 developed
  markets -- broadly confirming the Banz result internationally.
- **Asness, Frazzini, Israel & Moskowitz (2015).** "Fact, Fiction and Momentum Investing."
  *Journal of Portfolio Management*, 40(5). Contextualises which premiums are robust vs
  data-mined; the size premium survives with quality controls.
- **Hou & van Dijk (2019).** "Resurrecting the Size Effect: Firm Size, Profitability Shocks,
  and Expected Stock Returns." *Review of Financial Studies*, 32(7), 2850-2889. Argues the
  size premium was not dead in the 2000s-2010s once you control for profitability.

## The Magnificent-Seven era and winner-take-most dynamics

- **Autor, Dorn, Katz, Patterson & Van Reenen (2020).** "The Fall of the Labor Share and
  the Rise of Superstar Firms." *Quarterly Journal of Economics*, 135(2), 645-709. Argues
  that winner-take-most dynamics in technology markets create persistent concentration of
  profits and revenues among the largest firms -- the economic case for megacap persistence.
- **Philippon (2019).** *The Great Reversal: How America Gave Up on Free Markets.* Harvard
  University Press. Documents declining competition and rising market power among U.S.
  large-cap companies -- a structural argument for why the largest firms might maintain
  their position.
- **Bessembinder (2018).** "Do Stocks Outperform Treasury Bills?" *Journal of Financial
  Economics*, 129(3), 440-461. Shows that a handful of stocks account for most of the
  total wealth creation in the stock market -- the positive-skewness argument that favours
  holding the known winners. But note: identification of those winners is easier ex-post.

## Post-publication decay and data-mining concerns

- **McLean & Pontiff (2016).** "Does Academic Research Destroy Stock Return Predictability?"
  *Journal of Finance*, 71(1), 5-32. Documents that anomalies weaken after publication --
  the megacap concentration story has been heavily publicised since 2021, raising concern
  about forward-looking performance.
- **Harvey, Liu & Zhu (2016).** "...and the Cross-Section of Expected Returns." *Review of
  Financial Studies*, 29(1), 5-68. The multiple-comparisons problem in factor research:
  with hundreds of tested factors, a t-statistic of 2 is no longer sufficient for
  statistical significance. Our study tests k=7 and k=10 (two hypotheses); both fail even
  at t=2 on the full sample.

## Survivorship bias

- **Brown, Goetzmann, Ibbotson & Ross (1992).** "Survivorship Bias in Performance Studies."
  *Review of Financial Studies*, 5(4), 553-580. The canonical demonstration that studying
  only surviving funds/companies inflates estimated returns. Our universe (current S&P 500
  members) excludes all companies that were dropped from the index between 2008-2025 --
  including many that were dropped due to poor performance. All our returns are upper bounds.

## Method lineage

- **Newey & West (1987).** "A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix." *Econometrica* -- the HAC t-stat used
  in `strategy.summarize` and `quantlab.analytics.mean_tstat_hac`.
- **EDGAR fundamental data.** SEC EDGAR XBRL filings, concept
  `WeightedAverageNumberOfDilutedSharesOutstanding`, accessed via the desk's shared
  prefetch cache at `_cache/_edgar_*.parquet`. Market cap = shares x year-end close
  (yfinance `auto_adjust=True`).

## Related desk studies

- **[Study 45 -- Size-Premium](../../45-size-premium/):** the direct counterpart -- tests
  the Fama-French SMB factor. The history that the megacap-concentration story requires
  being dead.
- **[Study 68 -- All-Weather](../../68-all-weather/):** another allocation strategy that
  looks compelling in one era (2010s low-volatility) but disappoints over the full history.
- **[Study 102 -- Free-Rebalance](../../102-free-rebalance/):** annual rebalancing benefits
  across a diversified portfolio -- the comparative advantage of equal-weight over time.
- **[Study 50 -- High-Water](../../50-high-water/):** cross-sectional breadth and momentum --
  the complementary story to megacap concentration.
