# References & literature map — Study 79 (Sleigh-Ride)

## The claim under test

- **The original Hirsch observation.** Yale Hirsch, *Stock Traders Almanac* (1972 edition),
  first published the rule: *"If Santa Claus should fail to call, bears may come to Broad and
  Wall."* The window is the last 5 trading days of the calendar year plus the first 2 of the new
  year. Hirsch's son Jeffrey continues the almanac; the rally is now the most widely cited
  short-term calendar anomaly in popular finance. The follow-up claim — that a *negative*
  Santa window predicts a poor January and Q1 — is the directional forecast we test separately.

## Why the steelman is plausible — the real effects it leans on

- **Calendar and turn-of-year seasonality.** Rozeff & Kinney (1976), *Capital Market Seasonality:
  The Case of Stock Returns* (Journal of Financial Economics), documented the January Effect —
  above-average returns in the first month of the year, attributed to tax-loss selling followed
  by re-investment. The Santa window overlaps precisely with this re-investment period.
- **Window-dressing by institutional managers.** Portfolio managers sometimes buy winners into
  year-end to show strong holdings in annual reports, contributing to a year-end bid. Lakonishok,
  Shleifer, Thaler & Vishny (1991), *Window Dressing by Pension Fund Managers* (American Economic
  Review), and Haugen & Lakonishok (1988), *The Incredible January Effect*, document this
  mechanism. Our desk [Study 67 — Fed-Drift](../../67-fed-drift/) tests a similar event-window
  institutional-flow idea.
- **Reduced volume, thin markets.** The holiday period has lower trading volume, which can
  amplify any directional flow. Lo & Wang (2000), *Trading Volume: Definitions, Data Analysis,
  and Implications of Portfolio Theory* (Review of Financial Studies), link low-volume periods to
  higher per-share price impact.
- **Turn-of-year momentum.** Jegadeesh & Titman (1993), *Returns to Buying Winners and Selling
  Losers* (Journal of Finance), document short-term momentum; Grinblatt & Moskowitz (2004) find
  fiscal-year-end effects. The Santa window rides the interface of year-end selling and January
  re-entry momentum.

## Why it may not hold — the honest counter-case

- **Small-sample inference.** With only ~75 annual events (and only ~18 negative-window years to
  test the follow-up claim), the effective sample size is tiny. Fama (1991), *Efficient Capital
  Markets: II* (Journal of Finance), warns that calendar anomalies discovered in short histories
  are disproportionately likely to be data-snooped artefacts.
- **The January Effect has shrunk.** Gu (2003), *The declining January Effect: evidences from the
  U.S. equity markets* (Quarterly Review of Economics and Finance), and Bhardwaj & Brooks (1992)
  document the progressive attenuation of the January effect after its publication, consistent
  with the Schwert (2003), *Anomalies and Market Efficiency* (Handbook of the Economics of Finance)
  view that anomalies arbitraged away on discovery.
- **Just equity beta.** A buy-and-hold equity investor is already long over the Santa window and
  captures all ~121 bps of the gross return without any active management. The true *alpha* is
  only the excess above the baseline (~14.5 bps/day × 7 days ≈ +100 bps vs a passive hold).
  That positive-beta seasonal is not tradable alpha; it is compensation for holding equity risk
  over a 7-day window.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy._hac_tstat`](../sleigh_ride/strategy.py) and [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Random-window null.** The desk's standard baseline for event-window studies: draw from the
  empirical distribution of non-event windows of the same length. Applied in
  [`strategy.random_window_baseline`](../sleigh_ride/strategy.py).
- **Per-window log returns.** Standard in event-study methodology; see MacKinlay (1997),
  *Event Studies in Economics and Finance* (Journal of Economic Literature).
- **Block-bootstrap for calendar effects.** Politis & Romano (1994), *The Stationary Bootstrap*
  (JASA) — [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), auto-adjusted closes, ^GSPC since 1950
  (76 Santa windows, ~75 years of daily history) and SPY since 1993 (33 windows). Every headline
  number is pinned with an `as_of` date and a per-tape content fingerprint (see
  [`docs/results.md`](results.md)). The offline reproducible core and the test-suite run on the
  deterministic [`data.synthetic_daily`](../sleigh_ride/data.py) generator, never the network.

## Related desk studies

- **[Study 67 — Fed-Drift](../../67-fed-drift/)**: the pre-FOMC announcement drift — another
  event-window effect tied to institutional positioning, same HAC + random-window methodology.
- **[Study 71 — Ambush](../../71-ambush/)**: confluence of four seasonal and event-window edges
  — the desk's most recent calendar-family study.
- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: the daily golden cross — a rule-based signal
  on the same daily equity time-series, different family but the same verdict pattern.
- **[Study 70 — Digital-Gold](../../70-digital-gold/)**: crypto seasonality — the comparison desk
  study for whether calendar effects exist in non-equity asset classes.
