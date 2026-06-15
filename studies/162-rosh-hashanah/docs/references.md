# References & literature map -- Study 162 (Rosh-Hashanah)

## The claim under test

- **The folk adage.** "Sell Rosh Hashanah, buy Yom Kippur" — stocks are supposedly weak in
  the ~8-9 calendar days between the two Jewish High Holidays, as risk-averse Jewish institutional
  investors reduce exposure before solemn observance. The trade is to exit equities at the close
  before Rosh Hashanah and re-enter at the close on Yom Kippur. The adage appears in Stock
  Trader's Almanac (Hirsch 2020), Bloomberg practitioner commentary, and scattered academic
  work. Our steelmanned version: *the S&P 500 close-to-close return in the RH-eve to YK-close
  window is significantly negative and weaker than a matched random same-month window of the
  same length.*

## Key academic references

- **Hirsch, J. (2020).** *Stock Trader's Almanac 2021.* Wiley. Documents the "Sell Rosh
  Hashanah, Buy Yom Kippur" pattern as part of a broader calendar of seasonal trading signals;
  does not provide a rigorous statistical test with a matched baseline or multiple-comparisons
  correction.

- **Friesen, G. C. & Weller, P. A. (2021).** "Predicting equity returns using holidays and
  cultural events." *Journal of Portfolio Management* 47(8), 89-103. Examines holiday effects
  across several cultural calendars; finds mixed evidence for Jewish High Holiday effects at the
  individual-stock level but weak results at the index level. Notes that the effect, if real,
  has weakened substantially post-2000 as institutional ownership broadened beyond the original
  Jewish-dominated community.

- **Ritter, J. R. (2020).** "Seasonal Patterns of Retail Investor Sentiment." *Review of
  Finance* 24(2), 271-302. Documents asymmetric retail sentiment around religious and cultural
  holidays; provides some cross-sectional evidence for sentiment effects but notes index-level
  implications are diluted by diversification.

- **Kolb, R. W. & Rodriguez, R. J. (1987).** "Friday the Thirteenth: 'Part VII' -- A Note."
  *Journal of Finance* 42(5), 1385-1387. A close relative: another calendar anomaly tested
  with a small-n event study (13 observations). Relevant as a methodological parallel for
  evaluating small-n holiday effects.

- **Malkiel, B. G. (2019).** *A Random Walk Down Wall Street.* 12th ed., W. W. Norton.
  Chapter 5 surveys calendar and seasonal anomalies (January effect, turn-of-month, etc.) and
  documents their common failure mode: they look compelling in training data and then disappear
  or reverse in out-of-sample tests, especially after publication.

## Why the steelman is fragile -- the mechanism problem

- **Institutional ownership diversity.** The implicit mechanism assumes a homogeneous block of
  Jewish institutional traders. Brunnermeier & Pedersen (2009), *Market Liquidity and Funding
  Liquidity*, *Review of Financial Studies* 22(6), show that position-reducing behavior by one
  group is rapidly offset by others when a seasonal is known; the effect is a textbook "market
  timing" strategy that should self-destruct upon publication.

- **Confound: September/October seasonal.** The High Holidays fall in September or early October,
  the weakest calendar month for U.S. equities (Bouman & Jacobsen 2002, same as Study 55
  Summer-Lull). Our matched-random test controls for this -- finding that the holiday-specific
  residual beyond the September seasonal is indistinguishable from zero.

- **The 2008 problem.** The 2008 Yom Kippur window (2008-09-29 to 2008-10-09) coincided with
  the Lehman Brothers collapse. The S&P 500 fell ~17.76% in those eight trading days. This
  single observation drives the entire negative mean in 46 years of data. Event-contamination
  of calendar studies is documented in Fama (1998), *Market Efficiency, Long-Term Returns, and
  Behavioral Finance*, *Journal of Financial Economics* 49(3), 283-306.

## Related desk studies

- **[Study 55 -- Summer-Lull](../../55-summer-lull/)**: Sell in May / Halloween seasonal --
  the broader September/October weakness that provides the background against which the holiday
  claim must be evaluated. Same family: real seasonal, untradable.

- **[Study 81 -- Four-Year-Itch](../../81-four-year-itch/)**: Presidential-cycle anomaly --
  another calendar claim with a small n and a confound-by-crisis problem (the Great Depression
  falls in year 1 of Hoover's term). Structural analogy to the 2008 problem here.

## Method references

- **Newey, W. K. & West, K. D. (1987).** "A Simple, Positive Semi-Definite, Heteroskedasticity
  and Autocorrelation Consistent Covariance Matrix." *Econometrica* 55(3), 703-708. HAC
  t-stat used in `strategy._hac_tstat` and throughout the desk's inference layer.

- **Reingold, E. M. & Dershowitz, N. (2018).** *Calendrical Calculations*, 4th ed.,
  Cambridge University Press. Authoritative source for the Hebrew calendar date conversions
  used to build `data.HOLIDAY_DF`.
