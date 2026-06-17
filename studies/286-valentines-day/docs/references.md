# References & literature map — Study 286 (Valentines-Day)

## The claim under test

There is no peer-reviewed "Valentine's Day effect"; this is pure calendar folklore,
of the same genus as the Super Bowl Indicator and Mark Twain's "October is a
peculiarly dangerous month." The retail-press version runs roughly: *"The market is
in a good mood on Valentine's Day, so stocks tend to rise on Feb 14."* We test it as
a clean **single-day event study**: for each year 1950–2025, isolate the first ^GSPC
session on or after Feb 14 and ask whether its close-to-close return beats the
unconditional daily return.

## Why a single-day calendar effect is almost certainly a mirage

- **The base-rate / drift trap.** Equity indices have a small positive *daily* drift
  (≈3–4 bps for the S&P since 1950). A naive "is the day up?" test credits that drift
  to whatever date you picked. The correct null is the unconditional daily mean, not
  zero — exactly the mistake that inflates most "X-day effect" write-ups.

- **Power.** With 76 Valentine sessions and ~1%/day volatility, the standard error of
  the mean is ≈0.11%/day. A premium would need to be ≈0.23%/day to clear |t| = 2 —
  about six times the market's entire daily drift. No plausible "mood" mechanism
  produces a same-day index move of that size.

- **The day-of-week confound.** Because Feb 14 (and its roll-forward) lands on
  different weekdays across years, any apparent effect is entangled with the
  long-documented day-of-week seasonality (the "Monday effect"). We record the
  weekday of each session so this confound is visible.

- **Multiplicity / data snooping.** There are 365 calendar dates; testing the most
  charming one ex-post is a single draw from a very wide multiple-comparison net. A
  ~4 bps "edge" is exactly what the luckiest of 365 dates would show by chance.

## Academic literature on calendar effects and small-sample mirages

- **French, K. R. (1980).** "Stock Returns and the Weekend Effect." *Journal of
  Financial Economics*, 8(1), 55–69. The canonical day-of-week study; the relevant
  confound for any fixed-calendar-date effect.

- **Lakonishok, J. & Smidt, S. (1988).** "Are Seasonal Anomalies Real? A Ninety-Year
  Perspective." *Review of Financial Studies*, 1(4), 403–425. Shows most calendar
  anomalies are fragile out of sample and concentrated in specific sub-periods —
  precisely the 1950–87 vs 1988–2025 split we observe here.

- **Sullivan, R., Timmermann, A. & White, H. (2001).** "Dangers of Data Mining: The
  Case of Calendar Effects in Stock Returns." *Journal of Econometrics*, 105(1),
  249–286. The definitive demonstration that calendar effects vanish once you account
  for the number of calendar rules implicitly searched.

- **Harvey, C. R., Liu, Y. & Zhu, H. (2016).** "… and the Cross-Section of Expected
  Returns." *Review of Financial Studies*, 29(1), 5–68. Argues the appropriate t-stat
  hurdle for a "discovered" effect is ~3.0, not 2.0, after multiplicity. A single-day
  folklore effect with t ≈ 0.02 is not remotely in the conversation.

## Related desk studies (same calendar-folklore family)

- **[Study 158 — Super-Bowl](../../158-super-bowl/)**: the canonical "spurious omen"
  teardown; same base-rate trap, same tiny-n reckoning.
- **[Study 223 — Same-Month-Seasonality](../../223-same-month-seasonality/)**: the
  honest event-study / synthetic-control template this study mirrors.
- Other quirky-calendar teardowns on the desk: friday-13th, mercury-retrograde,
  pre-holiday, turn-of-month — all in the same genus.

## Method lineage

- **Event study.** One pre-specified session per year; the statistic is the mean of
  the 76 session returns versus the unconditional daily mean (the excess return).
- **One-sample t-test.** `scipy.stats.ttest_1samp` — once against 0 (to show the raw
  return is not even non-zero) and once against the unconditional mean (the honest
  test of the excess).
- **Newey-West (HAC) t-stat.** Bartlett-kernel long-run variance (lag 4) on the
  excess return — robust to the mild serial correlation in daily returns. This is the
  headline |t| ≥ 2 hurdle.
- **Permutation test.** Draw 76 random sessions from the same 1950–2025 window 10,000
  times; the two-sided p-value is the fraction of draws whose mean deviates from the
  unconditional mean by at least as much as the Valentine mean does.

## Data sources

- **^GSPC daily closes.** Yahoo Finance `^GSPC` daily prices (split-adjusted,
  price-only — no dividends), staged at `_cache/^GSPC_split_only.parquet`. The index
  level is back-revised by the vendor; named on the Signal axis.
- **Valentine session-date table.** Hardcoded in `data.py`. Derived once from the
  ^GSPC session calendar (the first trading day on or after Feb 14 each year, since
  Feb 14 often falls on a weekend or holiday), then frozen for offline reproducibility.
