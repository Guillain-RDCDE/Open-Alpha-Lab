# References & literature map — Study 81 (Four-Year-Itch)

## The claim under test

- **The Presidential Election Cycle.** The folklore: year 3 of a U.S. Presidential term
  (the pre-election year) is the best for equities because incumbents use fiscal and monetary
  policy to boost the economy before the vote, and years 1–2 are lean because the new
  administration spends political capital on unpopular reforms. The recipe is typically
  attributed to Yale Hirsch's *Stock Trader's Almanac*, which has documented the pattern since
  at least 1972. We steelman it as: *the mean annual S&P 500 return in year-3-of-term is
  meaningfully higher than the pooled return in years 1, 2, and 4, measured over all available
  cycles since 1928 (~24 observations).*

## The original academic claim

- **Hirsch, Y. (1972 onward).** *Stock Trader's Almanac.* The canonical popular source for the
  Presidential Cycle: "the pre-election year is the best year." Not peer-reviewed, but widely
  cited in the financial press and used as a timing signal by retail traders.
- **Nordhaus, W. D. (1975).** *The Political Business Cycle.* Review of Economic Studies, 42(2),
  169–190. The academic foundation: governments manipulate economic policy to generate expansions
  before elections — the mechanism the cycle story is built on.
- **Allvine, F. C. & O'Neill, D. E. (1980).** *Stock Market Returns and the Presidential Election
  Cycle.* Financial Analysts Journal, 36(5), 49–56. One of the first peer-reviewed studies
  documenting the four-year equity return cycle, finding year-3 outperformance.

## Why the pattern has theoretical grounds — but thin empirical footing

- **Tufte, E. R. (1978).** *Political Control of the Economy.* Princeton University Press. Provides
  broad evidence that administrations manipulate short-term economic outcomes before elections,
  giving the cycle mechanism real credibility in principle.
- **Huang, R. D. (1985).** *Common Stock Returns and Presidential Elections.* Financial Analysts
  Journal, 41(2), 58–61. Finds the cycle effect significant in the period studied but notes
  sensitivity to the sample window.
- **Hensel, C. R. & Ziemba, W. T. (1995).** *United States Investment Returns During Democratic
  and Republican Administrations, 1928–1993.* Financial Analysts Journal, 51(2), 61–69.
  Documents year-of-term patterns and party effects — useful for the multi-factor view. Finds
  the year-3 advantage concentrated in the early part of the sample.

## The honest critique — small n and data snooping

- **Sullivan, R., Timmermann, A. & White, H. (1999).** *Data-Snooping, Technical Trading Rule
  Performance, and the Bootstrap.* Journal of Finance, 54(5), 1647–1691. Formalises why a
  pattern identified over 24 non-overlapping data points on one index is almost certainly
  data-mined, especially when dozens of calendar effects are tested simultaneously (month of
  year, day of week, turn of month, election year …).
- **Siegel, J. J. (2014).** *Stocks for the Long Run* (5th ed.), McGraw-Hill. Chapter 17
  examines the Presidential Cycle and concludes the effect is suggestive but not statistically
  robust enough to guide allocation decisions — consistent with our Welch t = 1.92.
- **Lucey, B. M. & Zhao, S. (2008).** *Halloween or January? Yet Another Puzzle.* International
  Review of Financial Analysis, 17(5), 1055–1069. Illustrates how calendar effects compete for
  the same variance and are often jointly insignificant when tested together.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.* Econometrica — the
  inference method used in [`strategy._hac_tstat`](../four_year_itch/strategy.py). Applied
  per year-of-term slot on ~24 annual observations.
- **Welch two-sample t-test.** Welch, B. L. (1947), *The Generalization of Student's Problem
  when Several Different Population Variances Are Involved.* Biometrika — used for the
  year-3 vs rest comparison; appropriate because the four year-of-term slots have unequal
  variances.
- **Synthetic positive control.** The `data.synthetic_daily` generator plants a known
  year-of-term premium; the test suite confirms the engine detects it when present and reads
  near-zero when absent — so the real-tape "weak" verdict is a statement about the data, not a
  defect in the machinery.

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`): ^GSPC price index back to 1928-01-03 (~24,700
  trading days, 24 complete Presidential-term cycles). No dividends in the price index — the
  annual-return *differences* between years are unaffected by this. SPY (total-return ETF,
  post-1993) used for robustness check. Both tapes cached under `_cache/`; fingerprinted in
  [`docs/results.md`](results.md).

## Related desk studies

- **[Study 42 — Last-Call](../../42-last-call/)**: turn-of-the-month seasonal — the
  closest sibling in the calendar family; a real but small and largely-traded-away premium.
- **[Study 48 — Groundhog](../../48-groundhog/)**: another calendar anomaly with limited
  out-of-sample evidence — the data-snooping / small-n parallel.
- **[Study 79 — Sleigh-Ride](../../79-sleigh-ride/)**: the January / year-end seasonal,
  the most-studied calendar effect, also weak in modern data.
- **[Study 67 — Fed-Drift](../../67-fed-drift/)**: policy-driven equity drift around FOMC
  meetings — the same broad family (government policy → equity premium), better identified.
