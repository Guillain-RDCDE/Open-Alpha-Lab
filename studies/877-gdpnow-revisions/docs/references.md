# References & literature map — Study 877 (GDPNow Revisions)

## The claim under test

- **The nowcast.** The **Federal Reserve Bank of Atlanta**'s *GDPNow* model produces a
  running, model-based estimate of the current quarter's real-GDP growth, **updated 6–7 times
  a quarter** (in practice almost every business day near the end of a quarter) as each new
  monthly release — payrolls, ISM, retail sales, trade, construction, inventories — is folded
  in. See the Atlanta Fed's *GDPNow* page and Higgins, P. (2014), *"GDPNow: A Model for GDP
  'Nowcasting'"* (FRB Atlanta Working Paper 2014-7).
- **The believers' story.** Because the nowcast moves *in response to* hard data, its **daily
  revision** is a clean, real-time "growth surprise": an upward revision means the incoming
  data beat the model's running estimate, which — the claim goes — should push equity prices
  up over the next day or week, and a large **downward** revision should precede weakness.
  This is the nowcast-as-tradable-macro-signal thesis.
- **The specific test here.** We take the top-line GDPNow nowcast's **full daily forecast
  history** (2011–2026), form the **within-quarter day-over-day revision**, and run a
  predictive regression of the 1- and 5-trading-day forward SPY return on that revision, with
  a Newey-West *t* and *R²*, a top/bottom-decile conditional test, a two-era cut, a permutation
  placebo, a costed timer, and a seeded synthetic positive control. We deliberately act at the
  release-day close (the most generous execution) and show the result does not survive a
  one-day lag.

## Why the claim is fragile a priori

- **Nowcast revisions are, by construction, already public.** GDPNow updates are posted right
  after the same releases that markets trade in the first minutes — payrolls at 8:30, ISM at
  10:00. By the time the revised nowcast is on the Atlanta Fed's site, the price move is
  largely done. A daily-sampled revision is a *coincident restatement* of news the tape has
  already absorbed, not a forecast of tomorrow.
- **The signal is a slow, mechanical accumulator.** GDPNow blends bridge equations and a
  dynamic factor; its revisions inherit the autocorrelation of the underlying data flow, so a
  naive OLS would overstate significance — hence the HAC (Newey-West) *t* throughout.

## What we measure, and the honesty rails

- **Revision, within-quarter.** `groupby("Quarter being forecasted").diff()` on the top-line
  nowcast, so no revision bleeds across the quarter boundary (a new quarter's *initial*
  forecast is not a "revision"); the first forecast of each quarter is dropped.
- **One documented execution lag.** Headline = act at the **close of the release day** (lag 0).
  Because the nowcast posts intraday this is the *most favourable* assumption; we also report
  the stricter **next-day** lag (lag 1), under which the one significant sub-result flips sign.
- **Robust inference.** Newey-West (HAC, Bartlett, 5-lag) *t* on the slope and on each decile
  mean; a Welch *t* on the up-minus-down decile spread; a 5,000-draw permutation placebo that
  shuffles forward returns against revisions; a two-era robustness cut; a 20-seed synthetic
  positive control that plants (and recovers) a known revision→return edge.
- **The timer is graded separately.** A long/flat SPY rule after up-revisions, costed one-way
  per turn and raced against buy-and-hold on the same dates.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the slope and the decile means).
- **Wilson, E. B. (1927)** — score interval for a binomial share (shared inference kit).
- **Higgins, P. C. (2014)** — *GDPNow: A Model for GDP "Nowcasting"*, FRB Atlanta WP 2014-7.

## Data sources

- **Atlanta Fed GDPNow workbook** `GDPTrackingModelDataAndForecasts.xlsx`
  (`TrackingDeepArchives` + `TrackingArchives` sheets), top-line `GDP Nowcast` by forecast
  date, 2011-08-25 → 2026-06-26, cached under `_cache/gdpnow.csv`. Documented fallback: FRED
  daily series `GDPNOW`.
- **yfinance daily SPY** total-return close (`auto_adjust=True`), 2011 → 2026-06-30, cached
  under `_cache/spy.csv`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [387-economic-surprise-index](../../387-economic-surprise-index/) — a **Citi-style economic
  surprise index** built from six FRED series versus a *trailing-average* consensus, sampled
  **monthly**. This study is **not** that: it uses the Atlanta Fed's live **GDPNow nowcast**
  and its **daily revision** (a model's real-time restatement of one number, GDP), not a
  multi-series beat/miss composite against a trailing mean.
- [384-ism-pmi-regime](../../384-ism-pmi-regime/) — a **level/regime** signal on the ISM PMI
  (above/below 50). This study uses the **change (revision)** of a GDP nowcast, not a diffusion
  index's level regime.
- [268-sahm-rule](../../268-sahm-rule/) — a **recession-trigger** rule on the unemployment rate
  (a slow, rare, binary switch). This study is a **daily continuous** growth-revision signal,
  not a labour-market recession flag.
- [385-jobless-claims-momentum](../../385-jobless-claims-momentum/) — **momentum in initial
  jobless claims** (a single labour series' trend). This study is the **revision of a broad
  GDP nowcast**, a different input and a different (daily, not monthly) cadence.

None of the siblings sort on the **daily within-quarter revision of the Atlanta Fed GDPNow
nowcast** — this study's own axis.
