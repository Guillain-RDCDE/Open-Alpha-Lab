# References & literature map — Study 297 (March-Madness)

## The claim under test

There is no peer-reviewed paper claiming "the NCAA tournament moves the S&P 500
lower." The claim lives in the financial-media folklore ecosystem: every March,
op-eds and trading desks half-jokingly wonder whether the three-week national
basketball distraction shows up as thinner volume and softer returns. The honest
empirical question — does the index behave differently *inside* the tournament
window than *outside* it — is what this study tests. The window is hardcoded from
the bracket schedule (first-round Thursday through championship Monday, 1985-2025).

## Why a distraction effect is plausible-sounding — and why it almost certainly is not

- **The productivity-loss anchor.** Consulting/outplacement firm Challenger, Gray &
  Christmas publishes an annual estimate of lost workplace productivity during
  March Madness (figures in the low single-digit billions of dollars). These press
  releases are the cultural seed for the "distraction" idea. Lost desk-hours are
  not the same as a tradable mispricing: the marginal price-setter in S&P 500
  futures is not the office-pool participant streaming a 12-vs-5 upset.

- **Attention and asset prices.** There is a genuine academic literature on
  *attention* affecting prices, e.g. **Da, Engelberg & Gao (2011)**, "In Search of
  Attention," *Journal of Finance* 66(5), and **Barber & Odean (2008)**, "All That
  Glitters: The Effect of Attention on the Buying Behavior of Individual and
  Institutional Investors," *RFS* 21(2). These document attention effects on
  *individual* stocks and retail flow, not a calendar-wide index drag during a
  sports event.

- **Sentiment / mood and markets.** **Edmans, García & Norli (2007)**, "Sports
  Sentiment and Stock Returns," *Journal of Finance* 62(4), find that national
  team *soccer* losses are followed by next-day underperformance of the losing
  country's index — a sentiment channel. But that is an outcome (a loss) shock to a
  whole nation, not a three-week distraction window, and the effect is small and
  contested out of sample. **Kaplanski & Levy (2010)** on aviation disasters and
  the "sentiment" family share the same fragility.

- **Calendar-anomaly base rates.** **Lakonishok & Smidt (1988)**, "Are Seasonal
  Anomalies Real? A Ninety-Year Perspective," *RFS* 1(4), is the canonical warning:
  most calendar effects are tiny, period-specific, and vanish once you account for
  the number of windows implicitly searched. The tournament window overlaps the
  back half of March, a stretch with its own (weak, contested) seasonal stories.

## The data-snooping / small-n reckoning

- **Sullivan, Timmermann & White (1999)**, "Data-Snooping, Technical Trading Rule
  Performance, and the Bootstrap," *Journal of Finance* 54(5). Picking the most
  striking of many possible calendar windows ex-post inflates apparent
  significance; the right benchmark is a reality-check / permutation p-value.

- **Harvey, Liu & Zhu (2016)**, "... and the Cross-Section of Expected Returns,"
  *RFS* 29(1). With hundreds of candidate predictors searched across the
  literature, the appropriate t-stat hurdle for a new "anomaly" is ~3.0, not 2.0.
  A folklore distraction effect with only ~500 in-window days and a sub-bps mean
  gap does not approach even the lenient |t| = 2 bar.

## Method lineage

- **Newey-West HAC t-stat.** The mean daily return inside the window vs outside is
  tested with a heteroskedasticity-and-autocorrelation-consistent standard error
  (bandwidth = floor(4 (n/100)^(2/9))). REAL on this desk requires |t| >= 2 on the
  real tape; literature plausibility alone is at most WEAK.
- **Levene variance test.** The "madness = chaos" corollary is tested as a
  variance ratio (in/out) with a Levene equality-of-variance test.
- **Block / random-placement permutation.** Draw random subsets of trading days
  matching the tournament's ~5% calendar footprint and ask how often a random
  window's mean is at least as low as the real one — a distribution-free p-value.
- **Lagged, cost-charged avoidance strategy.** The tradability test holds cash in
  the window (one trading-day execution lag) and pays a one-way cost on each
  entry/exit; gross and net are both reported.

## Data sources

- **^GSPC daily (S&P 500 price index).** Split-adjusted daily OHLC staged at the
  repo-level `_cache/^GSPC_split_only.parquet`. We use close-to-close **price**
  returns (no dividends — labelled price-only on the Signal axis), 1985-2025, to
  match the 64+ team bracket era. Cache-first; the network is touched only on an
  explicit `fetch_daily(fetch=True)`.
- **NCAA tournament windows.** Hardcoded in `data.py` from NCAA.com bracket
  archives, Sports-Reference, and Wikipedia ("<year> NCAA Division I men's
  basketball tournament"). 2020 is intentionally absent — the tournament was
  cancelled (COVID-19), a documented gap rather than missing data.

## Related desk studies

- **[Study 158 — Super-Bowl](../../158-super-bowl/)**: the canonical sports-vs-market
  folklore teardown (same base-rate / small-n discipline).
- **[Study 164 — Mercury-Retrograde](../../164-mercury-retrograde/)**: the same
  in-window vs out-of-window daily-return machinery applied to an astrological
  "bad omen."
- **[Study 223 — Same-Month-Seasonality](../../223-same-month-seasonality/)**: the
  calendar-window discipline (sub-period stability, permutation nulls).
