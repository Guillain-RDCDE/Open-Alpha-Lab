# References & literature map — Study 285 (St-Patricks-Day)

## The claim under test

There is **no academic paper** asserting a "St. Patrick's Day effect." Unlike the
turn-of-the-month, the January effect, or the Halloween indicator, the
"wearin' o' the green" rally is pure market folklore: a fixed civil-calendar date
(March 17) that gets a green reputation in seasonal-trading commentary and almanac
chatter. We treat it as one more day-of-year candidate and hold it to the same bar
as any anomaly: a stable HAC t >= 2 that survives a placebo, a multiple-comparisons
correction, and a sub-period split.

## Why a single fixed date almost always looks "special"

- **The drift baseline.** US equities drift up ~+2.4 bps/day unconditionally. Any
  single calendar day will, on average, be positive — so "March 17 is usually green"
  is true of almost every day. The correct question is whether the day beats the
  drift by a statistically real margin, not whether it is merely positive.

- **Implicit multiple testing.** There are 365 calendar dates and dozens of named
  holidays. Searching them all and reporting the most striking one inflates apparent
  significance enormously. March 17 was chosen *because* of the holiday, not pre-
  registered, so the honest correction is a Bonferroni across the candidate days.

- **Regime luck.** A "century-long" pattern can be carried entirely by one lucky
  modern sub-sample. We split 1928–2026 into three regimes; an effect that lives in
  only one and reverses in the others is a regime artifact.

## Method lineage

- **Lakonishok, J. & Smidt, S. (1988).** "Are Seasonal Anomalies Real? A Ninety-Year
  Perspective." *Review of Financial Studies* 1(4), 403–425. The canonical reckoning
  for daily/seasonal calendar anomalies and the data-snooping problem — the
  intellectual backbone of this teardown.

- **Sullivan, R., Timmermann, A. & White, H. (1999).** "Data-Snooping, Technical
  Trading Rule Performance, and the Bootstrap." *Journal of Finance* 54(5),
  1647–1691. Formalises how mining many calendar rules inflates the apparent edge of
  the best one; motivates the Bonferroni day-of-March sweep here.

- **Newey, W. K. & West, K. D. (1987).** "A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
  *Econometrica* 55(3), 703–708. The HAC standard error we put on the event-day mean
  (Bartlett kernel, lag = floor(4·(n/100)^(2/9))).

- **Harvey, C. R., Liu, Y. & Zhu, H. (2016).** "… and the Cross-Section of Expected
  Returns." *Review of Financial Studies* 29(1), 5–68. Argues the appropriate t-stat
  hurdle for a *newly mined* predictor is ~3.0, not 2.0 — the St. Patrick bump does
  not clear even 1.2 in-sample.

## Related desk studies (same small-n / calendar-folklore family)

- **[Study 158 — Super-Bowl](../../158-super-bowl/)**: a fixed-event indicator with
  the same base-rate / tiny-n trap.
- **[Study 163 — Friday-13th](../../163-friday-13th/)**: a daily ^GSPC event study on
  a superstition date — the structural twin of this study, with a placebo and a
  day-of-month multiple-comparisons sweep.
- **[Study 161 — Year-Ending-Five](../../161-year-ending-five/)**: a "decennial
  pattern" carried by a handful of lucky years.

## Data sources

- **^GSPC daily close.** From the shared repo cache
  `_cache/^GSPC_split_only.parquet` (1927-12-30 onward), originally `yfinance`.
  Price-only log-returns (no dividends): a total-return tape would lift *every* day
  equally and would not change the St. Patrick-vs-baseline contrast. ^GSPC is an
  index, so there is **no survivorship selection** on the return series itself.
- **St. Patrick's Day sessions.** Derived by date arithmetic in `data.py` — the
  first trading day on or after March 17 of each year — not a transcribed table.
