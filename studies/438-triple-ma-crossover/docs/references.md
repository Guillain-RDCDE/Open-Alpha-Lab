# References & literature map — Study 438 (Triple-MA-Crossover)

## The claim's source — where "three beats two" comes from

The triple moving-average crossover is retail-trading folklore rather than an academic
claim. It is a staple of trading YouTube, TradingView scripts, and broker education pages:
the pitch is that stacking a third ("medium") MA between the fast and slow lines and
requiring `fast > medium > slow` acts as a *confirmation filter* that screens out the
whipsaws a plain two-MA cross suffers. Common parameter sets are 5/20/50 and 10/50/200; the
"guppy multiple moving average" (Daryl Guppy) and "MA ribbon" indicators generalise the
same idea to many lines.

- Investopedia, *Triple Moving Average Crossover* and *Moving Average Crossovers* — the
  canonical retail framing of the rule and its "confirmation" rationale.
- Guppy, D. — *Guppy Multiple Moving Average (GMMA)*, the many-line ribbon that popularised
  "more averages = more confirmation."

## Moving-average timing in the literature (the honest record)

- **Brock, Lakonishok & LeBaron (1992)**, *Simple Technical Trading Rules and the
  Stochastic Properties of Stock Returns*, Journal of Finance — the classic that found MA
  crossover rules predictive on the Dow, *before* costs.
- **Sullivan, Timmermann & White (1999)**, *Data-Snooping, Technical Trading Rule
  Performance, and the Bootstrap*, Journal of Finance — shows the Brock-et-al. rules largely
  evaporate once you correct for the universe of rules searched. The direct intellectual
  ancestor of this study's "race the simpler benchmark + permutation null" stance.
- **Faber, M. (2007)**, *A Quantitative Approach to Tactical Asset Allocation*, Journal of
  Wealth Management — the 10-month/200-day single-SMA timing rule. Its main, robust benefit
  is **drawdown reduction**, not Sharpe alpha — exactly what we find the third MA's
  "safety" reduces to (see desk study 110).
- **Zakamulin, V. (2017)**, *Market Timing with Moving Averages* (Springer) — a book-length
  out-of-sample audit concluding that the apparent superiority of elaborate MA schemes over
  simple ones is largely an artifact of in-sample selection; simpler rules are no worse.
- **Marshall, Nguyen & Visaltanachoti (2017)** and related work — MA timing edges shrink to
  insignificance net of realistic costs on liquid indices.

## Shared-method citations (the desk engine)

- **Newey & West (1987)** — heteroskedasticity-and-autocorrelation-consistent (HAC)
  standard errors; the *t*-stat on the excess mean and the Sharpe-difference test.
- **Jobson & Korkie (1981); Ledoit & Wolf (2008)** — testing the difference of two Sharpe
  ratios (here in its simplest Newey-West return-difference form).
- **Politis & Romano (1994)** — the circular/stationary block bootstrap behind the
  permutation null (block ≈ 21 days, to preserve short-horizon autocorrelation while
  destroying trend persistence).
- **White (2000)**, *A Reality Check for Data Snooping* — the multiple-rules correction
  whose spirit motivates racing the triple rule against the simpler two-MA benchmark.

## Related desk studies

- [`../../110-faber-timing`](../../110-faber-timing) — the single-SMA tactical rule; its
  benefit is drawdown reduction, not alpha (the same lesson, one MA).
- [`../../91-death-cross`](../../91-death-cross) — the 50/200 two-MA "death cross" / "golden
  cross", the dual-MA cousin of this study's benchmark leg.
- [`../../104-bollinger-reversion`](../../104-bollinger-reversion) — another classic-TA
  indicator raced against an honest null on the same engine idiom.
- [`../../106-supertrend`](../../106-supertrend) and
  [`../../103-turtle-trader`](../../103-turtle-trader) — trend-following rules that *do*
  clear the bar, for contrast with this one that doesn't.
- [`../../178-cci`](../../178-cci) — an oscillator teardown sharing the data/strategy idiom.
