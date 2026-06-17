# References & literature map — Study 287 (Easter-Effect)

## The claim under test

The retail-folklore "Easter effect" comes in two flavours: that the market is *upbeat
going into the long Easter weekend* (the pre-Good-Friday session) and *sluggish coming
back* (Easter Monday). Good Friday (= Easter Sunday − 2 days) is an NYSE holiday, so
there is no Good-Friday return to measure; we test the two **bracketing sessions** as a
clean event study, 1950–2025, against the unconditional ^GSPC daily return — and,
crucially, against the **same-weekday** baseline, because the pre-holiday session is
always a Thursday and Easter Monday is always a Monday.

## Why this one is different from the pure-folklore calendar studies

Most calendar-folklore (Super Bowl, Valentine's Day, Friday-13th) tests a *random* date
against the drift and finds nothing. The Easter case is different because the event
sits at the gate of a **market holiday**, and the **pre-holiday effect** is one of the
few calendar regularities with real academic support:

- **Ariel, R. A. (1990).** "High Stock Returns Before Holidays: Existence and Evidence
  on Possible Causes." *Journal of Finance*, 45(5), 1611–1626. The canonical
  pre-holiday paper: the trading day before exchange holidays earns many times the
  average daily return. Good Friday is one of the holidays studied.

- **Lakonishok, J. & Smidt, S. (1988).** "Are Seasonal Anomalies Real? A Ninety-Year
  Perspective." *Review of Financial Studies*, 1(4), 403–425. Documents the pre-holiday
  effect over 90 years and shows most *other* calendar anomalies are fragile — the right
  frame for "robust on the whole tape but uneven across sub-periods."

- **Kim, C.-W. & Park, J. (1994).** "Holiday Effects and Stock Returns: Further
  Evidence." *Journal of Financial and Quantitative Analysis*, 29(1), 145–157.
  International and out-of-sample evidence on the holiday effect and its persistence.

- **French, K. R. (1980).** "Stock Returns and the Weekend Effect." *Journal of
  Financial Economics*, 8(1), 55–69. The Monday effect — the confound we must remove
  before claiming anything about Easter Monday (we test against a Monday-only baseline).

## Why a real pre-holiday drift is still a fragile *trade*

- **Once a year.** The pre-holiday session fires a single time per year. Even a +30 bps
  gross event return compounds to only ~0.31%/yr — a rounding error in a portfolio that
  could instead be fully invested 252 days a year.

- **Post-publication decay.** The pre-holiday effect has been public since the 1980s and
  has weakened or reversed in several markets and recent decades (the standard fate of
  documented anomalies; cf. McLean & Pontiff 2016). Our middle third (1980–2002) is
  already weak (HAC t = 1.19).

- **No leverage on an overnight gap.** The edge lives in one session's close-to-close
  move; you cannot safely lever a once-a-year overnight position to make the sliver
  material without taking on tail risk that swamps it.

- **McLean, R. D. & Pontiff, J. (2016).** "Does Academic Research Destroy Stock Return
  Predictability?" *Journal of Finance*, 71(1), 5–32. The decay-after-publication result
  that makes any known calendar edge a candidate for FRAGILE rather than INVESTABLE.

## Related desk studies (same calendar / holiday family)

- **[Study 158 — Super-Bowl](../../158-super-bowl/)**: the canonical "spurious omen"
  teardown; the base-rate trap, the tiny-n reckoning.
- **[Study 285 — St-Patricks-Day](../../285-st-patricks-day/)** and
  **[Study 286 — Valentines-Day](../../286-valentines-day/)**: sibling single-day
  holiday event studies built in the same batch; both land None/Mirage. Easter differs
  because it brackets a *market holiday*.
- Other holiday/turn-of-period teardowns on the desk: pre-holiday, turn-of-month,
  window-dressing, turkey (Thanksgiving) — all in the same genus.

## Method lineage

- **Computus.** Easter Sunday is placed by the anonymous Gregorian algorithm
  (Meeus/Jones/Butcher) in `data.easter_sunday`; Good Friday = Easter − 2, Easter
  Monday = Easter + 1. The hardcoded session table is *self-checking* against this
  algorithm (a test asserts the frozen `easter` column equals `easter_sunday(year)`).
- **Event study.** Two pre-specified sessions per year (pre-holiday Thursday,
  post-holiday Monday); the statistic is the mean of the 76 returns versus the
  unconditional daily mean and the same-weekday mean (the excess return).
- **One-sample t-test.** `scipy.stats.ttest_1samp` — vs 0 and vs the baseline.
- **Newey-West (HAC) t-stat.** Bartlett-kernel long-run variance (lag 4) on the excess
  return. This is the headline |t| ≥ 2 hurdle.
- **Permutation test.** Draw 76 random sessions from the same window 10,000 times; the
  two-sided p-value is the fraction of draws whose mean deviates from the baseline by at
  least as much as the event mean does.

## Data sources

- **^GSPC daily closes.** Yahoo Finance `^GSPC` daily prices (split-adjusted,
  price-only — no dividends), staged at `_cache/^GSPC_split_only.parquet`. The index
  level is back-revised by the vendor; named on the Signal axis.
- **Easter / Good-Friday / Easter-Monday dates.** Computed by the Computus algorithm in
  `data.py`; the surrounding ^GSPC trading sessions are frozen once from the session
  calendar (Good Friday closed in all 76 years), then checked against the algorithm for
  reproducibility.
