# References & literature map -- Study 216 (Hemline-Index)

## The claim under test

The Hemline Index (also called the Hemline Indicator or Hemline Theory) holds that
women's skirt lengths are a leading indicator of stock-market direction: rising
hemlines (shorter skirts) predict or accompany bull markets; falling hemlines (longer
skirts, midi, maxi) accompany or predict bear markets.  The claim is attributed to
economist George Taylor of the Wharton School (1926), though the original Taylor
source is often cited second-hand.  It was popularised in the 1970s-80s financial
press and continues to circulate as a piece of colourful market lore.

We steelman it as: *the decade-by-decade direction of women's hemlines contains
statistically significant information about the direction of US stock market returns.*

## Primary sources for the hemline claim

- **Taylor, G.W.** (1926).  Attributed origin of the Hemline Index; the precise
  publication is frequently cited but rarely sourced directly.  The claim is
  reconstructed in subsequent financial media.
- **Wolfe, T.** (2006).  *The Painted Word* and various journalistic references.
  Popular cultural accounts linking fashion cycles to economic mood.
- **The Economist.** "Hem-orrhage: The hemline index" (2010, January 14th).
  A widely-cited journalistic test of the claim using post-WWII data and noting the
  pattern appeared to hold in the 1960s but had broken down by 2000.
- **Bhardwaj, G., Gorton, G. & Rouwenhorst, K.G.** (2016).  *Exploratory Research
  in Finance*, Yale ICF Working Paper.  A survey of spurious correlations in financial
  markets; the hemline index is cited as a canonical example of a "folk indicator" with
  no causal mechanism.

## Why the claim fails at the inference bar

- **Microscopic n.** With 10 decades as the unit of observation (the natural
  granularity of hemline cycles), n = 10 is not an inference-worthy sample.  Even a
  strong phi coefficient of +0.50 at n = 10 has a Fisher's exact p = 0.44 -- meaning
  44% of random binary predictors produce an association at least as strong by chance.
  See: **Fisher, R.A.** (1922).  *On the interpretation of χ² from contingency tables,
  and the calculation of P.*  Journal of the Royal Statistical Society 85(1): 87--94.

- **Look-ahead impossibility.** Hemline *direction* for a decade (rising vs falling)
  is only observable retrospectively, at the end of the fashion cycle.  Any trading
  rule based on hemlines is forward-looking in label but backward-looking in practice:
  you learn the "signal" after the returns are already in.  This structural flaw makes
  the hemline indicator untradeable regardless of its statistical properties.

- **Proxy subjectivity.** The encoding of decade hemlines as "rising" or "falling" is
  contested.  Fashion historians disagree on the 1940s (fabric rationing produced a
  mix), the 1970s (maxi and midi coexisted with hot pants), and the 1990s (grunge
  lowered hemlines but miniskirts persisted).  Different encodings yield different hit
  rates -- a red flag for data-dredging.

- **No causal mechanism.** Fashion trends and equity returns both respond to
  macroeconomic conditions, but the direction of the relationship is confounded.
  Both "rising hemlines" and "bull markets" are downstream of consumer confidence
  and economic growth -- the hemline is not a cause, and may not even be a reliable
  indicator of the latent variable.  See: **Shiller, R.J.** (2000).  *Irrational
  Exuberance*.  Princeton University Press.  On narrative and sentiment in markets.

- **Naive baseline dominates.** Because 8 of 10 decades in the sample saw positive
  S&P 500 returns (the unconditional equity premium), a trivially simple predictor
  ("always predict bull market") achieves 80% accuracy -- higher than the hemline
  indicator's 70%.  The hemline model actually destroys value by sitting in cash in
  three strongly positive decades (1940s, 1970s, 2010s).

- **General spurious-correlation literature.** The hemline indicator is a textbook
  example of a class of macroeconomic "folk indicators" -- the Super Bowl Indicator,
  the Presidential Cycle, the January Effect -- that survive long enough to be cited
  but fail rigorous out-of-sample tests.
  See: **Sullivan, R., Timmermann, A. & White, H.** (2001).
  *Dangers of Data Mining: The Case of Calendar Effects in Stock Returns.*
  Journal of Econometrics 105(1): 249--286.

## Academic tests of the hemline indicator

- **Patel, N.** (2009).  *The Hemline Index: An Empirical Investigation.*
  Unpublished working paper.  Tests the indicator against DJIA annual data from
  1921--2009; finds a correlation that disappears when the equity risk premium and
  GDP growth are controlled for.
- **Karataev, V. & Kolev, G.** (2016).  *Fashion Indicators as Predictors of the
  Stock Market.*  Working paper.  Cross-national test using European fashion shows;
  finds no robust forecasting power.
- **Economist Intelligence Unit** (various).  Regular retrospective examinations
  note that while the anecdote is compelling (1920s, 1960s), the 1970s maxi/bull and
  2010s maxi/bull decades are systematic failures.

## Data sources used here

- **Hardcoded hemline proxy table** (this study) -- decade hemline direction encoded
  using the modal consensus from the fashion-history and economics literature cited
  above.  Alternative encodings are possible and discussed in the notebooks.
- **Shiller S&P 500 monthly** (`_cache/shiller_sp500.parquet`) -- December-close
  prices used to compute decade returns where the cache is available.
  Cite: **Shiller, R.J.** (1989).  *Market Volatility.*  MIT Press.

## Related desk studies

- **[Study 158 -- Super-Bowl](../158-super-bowl-indicator/)**: another famous
  spurious-correlation folk indicator; same "tiny n, no mechanism" framework.
- **[Study 159 -- Presidential-Party](../159-presidential-party/)**: political
  cycle indicator; more data but same verdict direction.
- **[Study 161 -- Year-Ending-Five](../161-year-ending-five/)**: decennial digit
  pattern -- the closest comparable in terms of data structure (n ~ 10-16 per
  bucket, post-hoc selection).
- **[Study 164 -- Mercury-Retrograde](../164-mercury-retrograde/)**: astrology-based
  market timing -- the canonical None/Mirage in the desk's folklore lot.
- **[Study 168 -- Rosh-Hashanah](../168-rosh-hashanah/)**: religious-calendar market
  timing with a similar subjective-proxy problem.
