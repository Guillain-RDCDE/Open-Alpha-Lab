# References & literature map — Study 161 (Year-Ending-Five)

## The claim under test

The decennial-cycle folklore holds that calendar years whose last digit is 5 are
reliably the best for US stocks, and years ending in 0 are the worst.  The full
"decennial pattern" extends to all ten digits: years ending in 3, 4, 5 are good;
years ending in 0, 7 are bad.  The claim is widely cited in market-timing newsletters
and stock-almanac literature, most prominently in Hirsch's *Stock Trader's Almanac*
(various editions since 1968) and Ned Davis Research commentary.  We steelman it as:
*the last digit of the calendar year contains statistically significant information
about the S&P 500's full-year return, over and above the unconditional equity premium.*

## Primary sources for the decennial folklore

- **Hirsch, J.A.** and successors.  *Stock Trader's Almanac* (various editions,
  1968–present), Wiley.  The annual almanac tracks the decennial pattern prominently
  and coined the mnemonic "years ending in 5 are the best."  The underlying data
  table has been reproduced in thousands of financial-media articles.
- **Ned Davis Research.** Decennial pattern commentary and charts have circulated in
  institutional publications since at least the 1980s.  The NDR version emphasises
  the post-WWII Dow Jones data window.
- **LeBeau, C. & Lucas, D.W.** (1992).  *Computer Analysis of the Futures Market*.
  Irwin.  An early computerised examination of calendar seasonality including the
  decennial cycle.

## Why the steelman fails at the inference bar

- **Multiple comparisons / data-snooping.**  The decennial claim is derived by
  grouping 150+ years of data into 10 buckets and selecting the best.  Bonferroni
  correction multiplies any single-bucket p-value by 10; our permutation equivalent
  inflates by ~11x.  The corrected p (best-of-10 = 0.008) technically survives a 5%
  bar but only because the digit-5 effect is unusually large — and the claim is
  not pre-registered.
- **Small effective n.**  With n ≈ 15–16 per digit, a single outlier year (e.g.
  1935: +41%, 1945: +32%) exerts enormous influence.  Lo (2002) and standard
  power-analysis tables show that at n = 16 even a Sharpe ratio of 0.6 is barely
  detectable at 5% (power < 50%).  Claiming a pattern from 16 data points in a
  single pre-selected bucket is the epistemological definition of "pattern-in-noise."
  See: **Lo, A.W.** (2002).  *The Statistics of Sharpe Ratios.*  Financial Analysts
  Journal 58(4): 36–52.
- **No causal mechanism.**  There is no proposed mechanism by which the last digit of
  the Gregorian year (a base-10 counting convention with no physical, economic, or
  policy meaning) could affect equity returns.  The claim is purely correlational
  with zero mechanistic support.  Contrast with genuinely seasonal effects (e.g. the
  January effect, which has at least a tax-loss-harvesting story) — even those largely
  disappear out-of-sample.  See: **Thaler, R.H.** (1987).  *Anomalies: The January
  Effect.*  Journal of Economic Perspectives 1(1): 197–201.
- **Post-publication / out-of-sample decay.**  Calendar anomalies tend to weaken or
  reverse once widely known, as arbitrageurs trade against them.  See: **McLean, R.D.
  & Pontiff, J.** (2016).  *Does Academic Research Destroy Stock Return
  Predictability?*  Journal of Finance 71(1): 5–32.  The digit-5 "anomaly" has been
  widely cited since the 1980s; the 2005, 2015, 2025 returns (+5.2%, −0.0%, +14.0%)
  are notably weaker than the pre-1990 average.

## The broader seasonality literature

- **Kamstra, M.J., Kramer, L.A. & Levi, M.D.** (2003).  *Winter Blues: A SAD Stock
  Market Cycle.*  American Economic Review 93(1): 324–343.  A rigorous test of a
  calendar-driven anomaly (daylight-hours / seasonal affective disorder) that does
  find a measurable effect — the methodological standard the decennial cycle must clear
  but does not.
- **Sullivan, R., Timmermann, A. & White, H.** (2001).  *Dangers of Data Mining: The
  Case of Calendar Effects in Stock Returns.*  Journal of Econometrics 105(1): 249–286.
  The definitive paper on the data-snooping problem in calendar effects: after
  correcting for the universe of calendar rules examined, most "significant" effects
  disappear.  The decennial digit is exactly the kind of rule this paper targets.
- **Haugen, R.A. & Jorion, P.** (1996).  *The January Effect: Still There After All
  These Years.*  Financial Analysts Journal 52(1): 27–31.  Even the best-supported
  calendar effect (January) is weaker out-of-sample — illustrating the general pattern.

## Method lineage (the desk's shared engine)

- **Newey-West HAC t-stat.**  Newey, W.K. & West, K.D. (1987).  *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.*
  Econometrica 55(3): 703–708.  Used in `strategy._hac_tstat_vs_null`.
- **Permutation inference.**  Good, P.I. (2005).  *Permutation, Parametric and
  Bootstrap Tests of Hypotheses* (3rd ed.).  Springer.  The permutation test in
  `strategy.permutation_test` directly implements the "best-of-10" data-snooping
  correction described here.
- **Reproducibility stamp.**  Shiller S&P 500 monthly dataset (public, maintained at
  http://www.econ.yale.edu/~shiller/data.htm), staged at `_cache/shiller_sp500.parquet`
  in the shared repo cache.  Every headline run is pinned with a content fingerprint.

## Data sources used here

- **Shiller S&P 500 monthly** (`_cache/shiller_sp500.parquet`) — columns: Date,
  SP500, Dividend, Earnings, Consumer Price Index, Long Interest Rate, Real Price,
  Real Dividend, Real Earnings, PE10.  Monthly 1871–2026; December closes used for
  calendar-year return computation.  Cite: **Shiller, R.J.** (1989).  *Market
  Volatility.*  MIT Press.

## Related desk studies

- **[Study 48 — Groundhog](../../48-groundhog/)**: another calendar folklore claim,
  same "tiny n, no mechanism" framework.
- **[Study 80 — Cold-Open](../../80-cold-open/)**: January Barometer — a more credible
  seasonal claim with n ≈ 75, yet still WEAK/FRAGILE.
- **[Study 55 — Summer-Lull](../../55-summer-lull/)**: "Sell in May" — the most famous
  seasonal rule, tested with the same honest multiple-comparisons accounting.
- **[Study 136 — Mark-Twain](../../136-mark-twain/)**: October effect — another
  month-based calendar claim with a named urban legend behind it.
- **[Study 83 — Half-Life](../../83-half-life/)**: n=3 event study (Bitcoin halvings)
  — the canonical desk treatment of tiny-n inference failure.
- **[Study 81 — Four-Year-Itch](../../81-four-year-itch/)**: presidential cycle
  seasonality — another decadal/multi-year pattern test with small n.
