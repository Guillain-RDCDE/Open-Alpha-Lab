# References & literature map — Study 135 (FOMC-Cycle)

## The claim under test

- **Cieslak, Morse & Vuolteenaho (2019)**, "Stock Returns over the FOMC Cycle,"
  *Journal of Finance* 74(5), 2201–2248. The source paper. US equity returns since 1994
  accumulate in the even-numbered weeks (0, 2, 4) of the ~6-week FOMC inter-meeting
  cycle. Week 0 begins on the statement day; each week is five trading days. The authors
  attribute the pattern to informal Fed communication (the Fed's "back-channel" to
  primary dealers) and find it concentrated in the pre-FOMC-day portion of week 0. Their
  full-sample Welch-t on the even-vs-odd gap exceeds 2, but the sample spans 1994–2014.
  We test the same hypothesis on the extended 1994–2026 sample and find the gap has
  decayed post-publication.

## Why the steelman is plausible — the underlying mechanism

- **Federal Reserve informal communication.** Cieslak, Morse & Vuolteenaho (2019)
  hypothesise that the Fed communicates informally (through media contacts and primary
  dealer meetings) predominantly in even weeks, leading to a predictable risk-premium
  compression. The pre-FOMC announcement drift (Lucca & Moench 2015, Study 67) is the
  most visible tip of this iceberg.
- **Lucca & Moench (2015)**, "The Pre-FOMC Announcement Drift,"
  *Journal of Finance* 70(1), 329–371. Documents the 24-hour pre-announcement equity
  drift — the most studied sub-component of the CMV cycle. Study 67 in this repo tests
  that specific window and finds it has also substantially decayed since publication.
- **Scheduled vs unscheduled information.** Bernile, Bhagwat & Rau (2016), "What's in
  the News? Information Content of FOMC Announcements," *Journal of Financial Economics*
  122(1), 153–178: scheduled FOMC meetings carry more information than unscheduled; the
  market impact is concentrated in the announcement window.

## Post-publication decay evidence

- **McLean & Pontiff (2016)**, "Does Academic Research Destroy Stock Return
  Predictability?," *Journal of Finance* 71(1), 5–32. Document that anomaly returns
  shrink by ~26% after journal publication and ~58% after working-paper circulation, as
  arbitrageurs learn about and trade away the pattern. The FOMC cycle pattern was widely
  circulated as a working paper from 2014 onward; our post-2019 gap is −1.72 bps/day
  (reversed), consistent with strong post-publication decay.
- **Chordia, Subrahmanyam & Tong (2014)**, "Have Capital Market Anomalies Attenuated in
  the Recent Era of High Liquidity and Trading Activity?," *Journal of Accounting and
  Economics* 58(1), 41–58: calendar anomalies tend to weaken as markets become more
  liquid and informationally efficient.

## Method lineage (the desk's shared engine)

- **Newey-West HAC t-stat.** Newey & West (1987), *Econometrica* 55(3), 703–708 —
  `strategy._hac_tstat` and [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
  Used on each week-arm mean to handle the serial correlation in daily equity returns.
- **Welch t-test.** Welch (1947), *Biometrika* 34(1/2), 28–35 — used for the
  even-vs-odd comparison (unequal variances and unequal sample sizes). The pooled-SE
  version would understate the t on this comparison.
- **Random permutation placebo.** Randomising the even/odd label allocation while
  preserving the true fraction of even-week days — a standard label-shuffle null for
  calendar effects.
- **Rolling decay.** Rolling Sharpe as in [`quantlab.analytics.rolling_sharpe`](../../../quantlab/analytics.py);
  the pre/post-publication era split follows McLean & Pontiff (2016).

## Data sources

- **SPY daily total returns** (via `yfinance`), 1994-01-03 to 2026-06-11, loaded from
  the shared repo cache at `_cache/last_call_spy.parquet`. The full sample gives
  n = 8,141 trading days with valid cycle-week labels.
- **FOMC meeting dates.** Federal Reserve Board historical FOMC calendars
  (https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm and archived
  meeting minutes pages). Hardcoded in `data.FOMC_DATES`, covering 1994–2026.

## Related desk studies

- **[Study 67 — Fed-Drift](../../67-fed-drift/)**: the pre-FOMC announcement drift —
  the specific 24-hour window that is the most visible sub-component of the CMV cycle.
  Tests the same mechanism one level more granularly; also finds strong pre-publication
  evidence and significant post-publication decay.
- **[Study 48 — Groundhog](../../48-groundhog/)** and
  **[Study 55 — Summer-Lull](../../55-summer-lull/)**: other calendar-driven return
  patterns tested with the same honest pre/post-publication split methodology.
- **[Study 82 — Witching-Hour](../../82-witching-hour/)**: another Fed/macro event
  calendar study showing similar decay dynamics.
