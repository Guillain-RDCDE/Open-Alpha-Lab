# References & literature map — Study 234 (Olympic-Year)

## The claim under test

The "Olympic-year effect" asserts that US equity markets perform better in years
hosting the Summer Olympic Games.  The intuition (to the extent any exists) rests on
vague notions of "global optimism," "international goodwill," "consumer sentiment,"
or "host-nation economic stimulus."  The claim has circulated in financial media as
a curiosity and is occasionally cited in stock-market-almanac-style commentary.  We
steelman it as: *the Summer Olympic Games calendar contains statistically significant
information about the S&P 500's full-year return.*

## Popular sources and the folklore

- **Various financial media (2012, 2016, 2020).** Articles in CNBC, MarketWatch, and
  similar outlets periodically run "Olympic effect" pieces, noting that "most Olympic
  years have been good for the stock market."  These articles typically cherry-pick
  the post-WWII era and a subset of major indices without statistical testing.  No
  peer-reviewed academic paper has proposed the Olympic-year effect as a serious anomaly.
- **Stock-market almanac tradition.** Calendar effects are a staple of retail
  financial almanacs (cf. Hirsch's *Stock Trader's Almanac*, various editions).
  The Olympic-year effect is a minor entry in this tradition, weaker and less
  frequently cited than the presidential cycle, January effect, or "year ending in 5."

## Why the steelman fails

- **n too small.**  Since 1928, there have been only 23 Summer Olympics in the data
  window (WWII cancelled three games).  With n = 23, the power of any test to
  detect a return premium below ~8 pp is essentially zero.  The structural floor
  is 4.3 independent events per decade.  See: **Lo, A.W.** (2002).  *The Statistics
  of Sharpe Ratios.*  Financial Analysts Journal 58(4): 36–52.
- **No causal mechanism.**  The IOC schedule is set years in advance; it contains no
  new information about macroeconomic conditions or corporate earnings.  The "optimism"
  story is untestable and post-hoc.  Compare: even the January effect — which has a
  plausible tax-loss-harvesting mechanism — largely disappears out-of-sample.
  See: **Thaler, R.H.** (1987).  *Anomalies: The January Effect.*  Journal of
  Economic Perspectives 1(1): 197–201.
- **Pattern reverses post-1972.**  The pre-1972 contrast (+3.5 pp) is positive;
  the post-1972 contrast (−0.9 pp) is negative.  This is the expected footprint of
  pure noise in small samples, not a persistent signal.
- **Data snooping / look-elsewhere.**  The Olympic-year effect is one of hundreds of
  calendar rules that can be constructed from the Gregorian calendar, election cycles,
  sports schedules, and cultural events.  Testing any single rule from this universe
  without pre-registration inflates the false-discovery rate dramatically.  See:
  **Sullivan, R., Timmermann, A. & White, H.** (2001).  *Dangers of Data Mining:
  The Case of Calendar Effects in Stock Returns.*  Journal of Econometrics 105(1):
  249–286.
- **Post-publication decay (general).**  Calendar anomalies weaken after public
  recognition.  See: **McLean, R.D. & Pontiff, J.** (2016).  *Does Academic Research
  Destroy Stock Return Predictability?*  Journal of Finance 71(1): 5–32.

## The broader seasonality literature

- **Kamstra, M.J., Kramer, L.A. & Levi, M.D.** (2003).  *Winter Blues: A SAD Stock
  Market Cycle.*  American Economic Review 93(1): 324–343.  A seasonal effect with
  a genuine mechanism (daylight hours / seasonal affective disorder) — the bar the
  Olympic claim must clear but does not.
- **Jacobsen, B. & Visaltanachoti, N.** (2009).  *The Halloween Effect in U.S. Sectors.*
  Financial Review 44(3): 437–459.  A methodologically careful test of seasonal
  effects in equity sectors.
- **Ferson, W.E. & Harvey, C.R.** (1993).  *The Risk and Predictability of
  International Equity Returns.*  Review of Financial Studies 6(3): 527–566.
  Evidence on cross-country return predictability; the Olympic host-nation channel
  (another version of the claim) is not supported in this literature.

## Olympic Games — data source

- **International Olympic Committee (IOC).**  Official list of Summer Olympic Games
  and host cities.  Available at https://olympics.com/ioc/summer-olympic-games.
  The hardcoded year table in `data.py` follows the IOC official schedule.  Games
  cancelled in 1916, 1940, 1944 (World Wars); the 2020 Tokyo Games were held in 2021
  (COVID-19 postponement) and are recorded under 2020 (the scheduled year) in the
  pre-announced calendar.

## Method lineage (the desk's shared engine)

- **Newey-West HAC t-stat.**  Newey, W.K. & West, K.D. (1987).  *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.*
  Econometrica 55(3): 703–708.  Used in `strategy._hac_tstat_vs_null`.
- **Permutation inference.**  Good, P.I. (2005).  *Permutation, Parametric and
  Bootstrap Tests of Hypotheses* (3rd ed.).  Springer.  The permutation test in
  `strategy.permutation_test` permutes the Olympic label across all years and
  measures how often a random labelling produces a contrast at least as extreme.
- **Data source.**  yfinance ^GSPC daily prices, December-close to December-close
  calendar-year log-returns.  Cached locally at `_cache/gspc_annual.parquet`.

## Related desk studies

- **[Study 161 — Year-Ending-Five](../../161-year-ending-five/)**: decennial digit
  cycle — another post-hoc calendar pattern, same "no mechanism" diagnosis.
- **[Study 159 — Presidential-Party](../../159-presidential-party/)**: presidential
  party cycle — a more credible structural claim with n ≈ 25, still WEAK.
- **[Study 174 — First-Five-Days](../../174-first-five-days/)**: January's first five
  days as a full-year predictor — same tiny-n folklore framework.
- **[Study 164 — Mercury-Retrograde](../../164-mercury-retrograde/)**: astrological
  calendar effect, definitionally no mechanism, None/Mirage.
- **[Study 168 — Chinese-Zodiac](../../168-chinese-zodiac/)**: another spurious
  calendar grouping with n ≈ 12 per animal sign.
