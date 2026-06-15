# References & literature map — Study 158 (Super-Bowl)

## The claim under test

**Krueger, A. B. & Kennedy, J. M. (1990).** "An Analysis of the Super Bowl
Stock Market Predictor." *The Journal of Finance*, 45(2), 691–697.
The canonical academic treatment of the indicator. Krueger and Kennedy document
that from Super Bowl I (1967) through Super Bowl XXIII (1989), the S&P 500 rose
in every year an "original NFL" team won and fell in most AFL/AFC years —
a 23-for-23 streak that generated breathless commentary. They note explicitly
that the sample is tiny, the mechanism is absent, and the effect is a coincidence.
Their punchline: "*the predictive power... is due entirely to the fact that the
market tends to rise in most years, combined with the NFC's dominance in early
Super Bowls.*"

## Why the streak looked so impressive — and why it broke

- **The base-rate trap.** The S&P rises in roughly 70–75% of calendar years
  unconditionally. A signal that predicts "up" for NFC years gets this for free.
  The correct null is binomial(n, p = 0.73), not binomial(n, p = 0.5). Tested
  correctly, even the original 1967–1997 streak is not statistically significant.

- **Sample-size problem.** With 59 total Super Bowls (~30 per conference) and
  annual S&P volatility of ~17%, the minimum detectable mean-return difference
  at |t| = 2 is approximately 6%/yr. The actual NFC vs AFC gap is ~2–3%/yr —
  undetectable at this sample size.

- **The NFC dynasty era.** NFC teams dominated Super Bowls from 1967–1997
  (Dallas Cowboys, San Francisco 49ers, Washington Redskins, NY Giants, Chicago
  Bears, Green Bay Packers). This era overlapped with a secular bull market,
  making the correlation entirely spurious. After 2001, with the New England
  Patriots (AFC) winning six Super Bowls during a mixed market era, the
  indicator's "success rate" collapsed from ~90% to ~55%.

- **Data snooping.** The Super Bowl Indicator was discovered *after* the streak.
  There are many other sports events, weather patterns, and cultural curiosities
  that would have shown similarly impressive backtests over 23 years. Picking
  the most striking one ex-post inflates its apparent significance.

## Academic literature on spurious correlations and small-n mirages

- **Leinweber, D. J. & Segre, A. M. (1996).** "The Financial Data Finder." In
  *Practical Risk-Adjusted Performance Measurement*. The authors famously showed
  that butter production in Bangladesh was more correlated with the S&P 500 than
  almost any financial variable — the Super Bowl Indicator belongs to this class.

- **Sullivan, R., Timmermann, A. & White, H. (1999).** "Data-Snooping, Technical
  Trading Rule Performance, and the Bootstrap." *Journal of Finance*, 54(5),
  1647–1691. A rigorous treatment of how data-mining inflates apparent predictive
  power; the correct benchmark is a Reality Check p-value that accounts for the
  number of indicators implicitly searched.

- **Harvey, C. R., Liu, Y. & Zhu, H. (2016).** "... and the Cross-Section of
  Expected Returns." *Review of Financial Studies*, 29(1), 5–68. Documents that
  the literature has produced hundreds of "significant" predictors, most of which
  are spurious; the appropriate t-stat hurdle for a new anomaly is now ~3.0, not 2.0.

## Related anomalies that share the same structure

- **Santa-Clara, P. & Valkanov, R. (2003).** "The Presidential Puzzle: Political
  Cycles and the Stock Market." *Journal of Finance*, 58(5), 1841–1872.
  Presidential party affiliation and stock returns — the same structure as the
  Super Bowl Indicator but with a slightly larger n (~19 terms). The authors find
  a real-looking effect but acknowledge the tiny n and possible confounds.

- **Hirsch, Y. (1967).** *Stock Trader's Almanac*. The Presidential Election Cycle
  — year 3 is the best for stocks — a related calendar anomaly. See
  [Study 81 — Four-Year-Itch](../../81-four-year-itch/) for the honest teardown.

## Method lineage

- **Binomial test.** The correct test for a hit-rate claim: `scipy.stats.binomtest`
  with p₀ equal to the *unconditional* up-rate, not 0.5.
- **Permutation test.** Shuffle conference labels 10,000 times; record the
  empirical distribution of hit-rates; the p-value is the fraction of shuffles
  that equal or exceed the observed hit-rate.
- **Welch t-test.** `scipy.stats.ttest_ind(equal_var=False)` on per-year returns
  — appropriate when group sizes differ slightly (~30 vs ~29).
- **Bonferroni correction.** Multiply p-values by the number of simultaneously
  tested hypotheses (2 here: NFC/conference and orig_nfl). Almost irrelevant
  given both tests are far from significance.
- **HAC / Newey-West** is not applicable here: with only 1 observation per year
  and 59 total annual observations, the standard Welch t already uses the
  correct degrees of freedom. Annual returns have negligible serial correlation.

## Data sources

- **Shiller S&P 500 monthly dataset.** Robert Shiller's long-run US equity data,
  staged at `_cache/shiller_sp500.parquet`. We use December-to-December nominal
  price returns for 1967–2025 to get 59 calendar-year returns matching the 59
  Super Bowls. No dividends (price return only, consistent with Krueger-Kennedy).
- **Super Bowl results table.** Hardcoded in `data.py`. Sources:
  Pro Football Reference (https://www.pro-football-reference.com/super-bowl/),
  Wikipedia "List of Super Bowl champions," and the original Krueger-Kennedy (1990)
  paper for the original-NFL vs AFL classification.

## Related desk studies

- **[Study 81 — Four-Year-Itch](../../81-four-year-itch/)**: Presidential election
  cycle — same family of calendar/political anomalies with an equally tiny n.
- **[Study 48 — Groundhog](../../48-groundhog/)**: Punxsutawney Phil's shadow vs
  S&P returns — another seasonal folklore indicator.
- **[Study 83 — Half-Life](../../83-half-life/)**: The n=tiny teardown methodology
  applied to other small-sample "anomalies."
