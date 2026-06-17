# References & literature map — Study 217 (Lipstick Index)

## The claim under test

**Lauder, L. A. (2001).** Anecdotal claim made in a media interview following
the September 2001 recession concerns. Leonard Lauder (then Chairman of Estee
Lauder Companies) observed that lipstick sales rose during the 2001 downturn
and suggested this reflected consumer "trade-down" from luxury goods to
affordable luxuries. The story spread rapidly in financial media and the
indicator was attributed to him personally, though it was never published in
any peer-reviewed venue. There is no formal model behind it.

## Economic rationale (steelmanned)

The claim rests on two mechanisms:

1. **Affordable luxury / trade-down.** During downturns, consumers cut large
   discretionary purchases (cars, vacations, luxury handbags) but compensate
   psychologically with small treats — lipstick, nail polish, cosmetics.
   Demand for affordable luxury is counter-cyclical.

2. **Defensive consumer staples tilt.** Cosmetics companies (Estee Lauder,
   Coty) straddle Consumer Staples and Consumer Discretionary. In recessions,
   consumer staples tend to outperform cyclicals.

Even if mechanism 1 were real, mechanism 2 (defensive tilt) is already
well-documented and priced — it does not generate alpha over the broad market.

## Academic context

- **Nystrom, P. H. (1929).** *Economics of Fashion*. Ronald Press. Early
  academic treatment of fashion cycles and economic conditions — shows the
  relationship between consumption patterns and the business cycle is complex
  and not monotonic.

- **Durante, R., Gutierrez, E. & Villamizar-Villegas, M. (2023).** "The
  Lipstick Effect: How Crisis Shapes Consumer Behavior." Working paper
  examining whether cosmetics consumption is genuinely counter-cyclical across
  countries. Finds mixed evidence with strong regional variation.

- **Ang, A., Hodrick, R. J., Xing, Y. & Zhang, X. (2006).** "The Cross-Section
  of Volatility and Expected Returns." *Journal of Finance*, 61(1), 259–299.
  Background on defensive vs cyclical equity characteristics; relevant for
  understanding why cosmetics might seem "defensive" but cannot beat market
  risk-adjusted returns on that basis alone.

- **Fama, E. F. & French, K. R. (1993).** "Common Risk Factors in the Returns
  on Stocks and Bonds." *Journal of Financial Economics*, 33(1), 3–56.
  The workhorse framework: sector-level defensive characteristics are captured
  by risk factors, not by unexploitable alpha.

- **NBER Business Cycle Dating Committee.** Official recession chronology used
  for this study. https://www.nber.org/research/business-cycles

## Media coverage that perpetuated the myth

- **Weil, J. (2009).** "Lipstick Index Still Glossing Over the Truth." Bloomberg.
  Notes that Estee Lauder's own data contradicted the index during the GFC.

- **Rozhon, T. (2003).** "The Lipstick Wars." *New York Times*. Traces the
  origins of the claim and notes it was never rigorously tested.

- **Various financial media, 2001–2020.** The indicator was widely cited as
  confirmed during the dot-com bust, then challenged during the GFC when
  cosmetics sales declined with the rest of consumer spending.

## Related desk studies

- **[Study 81 — Four-Year-Itch](../../81-four-year-itch/)**: Calendar/political
  anomaly — same structure of small n and convenient narrative.
- **[Study 158 — Super-Bowl](../../158-super-bowl/)**: The canonical spurious
  indicator study — demonstrates how compelling streaks arise from base-rate neglect.
- **[Study 95 — Holiday Cheer](../../95-holiday-cheer/)**: Another calendar
  "defensive" claim tested against the honest baseline.

## Method lineage

- **Welch t-test.** `scipy.stats.ttest_ind(equal_var=False)` on monthly relative
  returns (cosmetics - SPY) split by NBER recession indicator.
- **Permutation test.** Shuffle recession flags 10,000 times; empirical p-value
  is the fraction of shuffles where the mean-recession-minus-expansion gap
  equals or exceeds the observed value.
- **Data.** yfinance monthly total returns (auto-adjusted for splits/dividends).
  Equal-weight cosmetics basket of available tickers per month.

## Data sources

- **yfinance.** EL (Estee Lauder, 1996–), ULTA (Ulta Beauty, 2007–), COTY
  (Coty Inc., 2013–), ELF (e.l.f. Beauty, 2016–), SPY (S&P 500 ETF, 1994–).
  Cached at `studies/217-lipstick-index/_cache/cosmetics_monthly.parquet`.
- **NBER recession dates.** Hardcoded in `data.py`. Source:
  https://www.nber.org/research/business-cycles
