# References & literature map — Study 322 (FOMC-Blackout)

## The claim under test

- **The FOMC communications "blackout" / quiet period.** Federal Reserve Board, *Policy
  on External Communications of Committee Participants* — by rule, FOMC participants and
  senior staff refrain from public commentary on monetary policy from the *second
  Saturday before* a meeting through the *Thursday after* it (a ~10-day quiet period). The
  folk claim circulating among traders is that this pre-meeting silence produces a "calm
  before the storm" — an unusually steady, positive drift while the market waits for the
  decision, with the volatility concentrated in the announcement itself rather than the
  run-up. We test whether the blackout window carries an **excess** return over the rest
  of the year and whether it is genuinely calmer (lower volatility).

## The real, adjacent effect — the pre-FOMC announcement drift

- **Lucca & Moench (2015), *The Pre-FOMC Announcement Drift*** (Journal of Finance 70:1).
  The canonical finding: US equities earn a large, statistically significant excess return
  in the **24 hours before** scheduled FOMC announcements (~49 bps over 1994–2011),
  accounting for a sizeable share of the equity premium. This is a *narrow, one-day*
  window the day before the meeting — a different object from the ~10-day blackout. Our
  blackout test contains this single day but dilutes it across the whole quiet period; we
  report the one-day pre-FOMC drift separately so the two are not conflated. The blackout
  framing only "works" by borrowing this much narrower, well-documented effect.

- **Cieslak, Morse & Vuolteenaho (2019), *Stock Returns over the FOMC Cycle*** (Journal of
  Finance 74:5). The even/odd-week inter-meeting cycle, concentrated on the days *after*
  each statement. This is the subject of **[Study 135 — FOMC-Cycle](../../135-fomc-cycle/)**;
  Study 322 deliberately tests the **pre-meeting blackout window** instead, a distinct
  calendar object, to avoid re-running the same anomaly.

## Why an excess "blackout drift" should be doubted

- **Post-publication decay.** McLean & Pontiff (2016), *Does Academic Research Destroy
  Stock Return Predictability?* (Journal of Finance 71:1) — anomaly returns shrink by
  ~58% out of sample after publication. If a tradable blackout drift existed, the
  post-2011 era (after Lucca-Moench) should show decay; our pre/post-2011 split tests
  exactly that.
- **Conditioning on a positive-drift asset.** Any subset of trading days in a rising index
  will show a positive mean. The relevant statistic is the **difference** between blackout
  and non-blackout days, not the blackout level — a window that merely earns the same
  equity drift as everything else is "being long the market," not an edge. This is the
  desk's "normalise before you marvel" rule.
- **Calendar placebo.** Sliding the window off the meeting date and re-measuring is a
  cheap falsification: a real blackout effect peaks at the true window and fades as the
  window slides away; a flat profile across offsets says the label carries no information.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  implemented in [`strategy.hac_tstat`](../fomc_blackout/strategy.py).
- **Welch two-sample t.** Welch (1947), *The Generalization of Student's Problem when
  Several Different Population Variances are Involved* (Biometrika) — the blackout-minus-
  other difference test.
- **Excess-of-cash Sharpe race.** Both the blackout timing book and buy-and-hold are
  measured net of the same cash rate before comparison ([`strategy.excess_sharpe`](../fomc_blackout/strategy.py)),
  per the desk's house rule on apples-to-apples races.

## Data sources used here

- **Yahoo! Finance daily bars** for SPY, auto-adjusted (a total-return proxy), 1994–2026,
  pinned with an as-of date and content fingerprint (see [`docs/results.md`](results.md)).
  Loaded cache-first from the shared repo `_cache/SPY_total_return.parquet`. The offline
  reproducible core and the test-suite run on the deterministic
  [`data.synthetic_blackout`](../fomc_blackout/data.py) generator, never the network.
- **FOMC scheduled decision dates, 1994–2026** — Federal Reserve FOMC historical
  calendars, hardcoded in [`data.FOMC_DATES`](../fomc_blackout/data.py).

## Related desk studies

- **[Study 135 — FOMC-Cycle](../../135-fomc-cycle/)**: the Cieslak even/odd-week
  *post*-meeting drift. The sibling FOMC-calendar study; Study 322 is the *pre*-meeting
  blackout-window counterpart, kept deliberately distinct.
