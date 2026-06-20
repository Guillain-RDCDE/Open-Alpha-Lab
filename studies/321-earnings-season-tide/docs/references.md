# References & literature map — Study 321 (Earnings-Season-Tide)

## The claim under test

- **The "earnings-season tide" folk idea.** A recurring market-commentary trope: that the
  *whole* index behaves differently during the few weeks each quarter when the bulk of
  large-cap constituents report — mid-late January, April, July and October. Stated at full
  strength: the aggregate market is pulled by a distinct seasonal drift during peak earnings
  weeks, so you could lean on it by being long (or flat) only inside those windows. We test
  the strongest version directly: a pre-declared, hard-coded four-window calendar contrast on
  a broad-market total-return index, with a robust *t* on the in-window vs out-of-window
  difference. This is a calendar-known rule, so no execution lag applies.

## The real micro-effect this leans on — single-stock earnings dynamics

- **Post-earnings-announcement drift (PEAD).** Ball & Brown (1968), *An Empirical Evaluation
  of Accounting Income Numbers* (Journal of Accounting Research); Bernard & Thomas (1989),
  *Post-Earnings-Announcement Drift: Delayed Price Response or Risk Premium?* (Journal of
  Accounting Research). The robust *single-stock* anomaly — prices drift in the direction of
  the earnings surprise for weeks. It is the seed of the folk belief, but it is a
  *cross-sectional, surprise-conditioned* effect; it does **not** imply a directional
  *market-wide* calendar tide, because surprises are roughly symmetric in aggregate.
- **The earnings-announcement premium.** Beaver (1968) and Frazzini & Lamont (2007), *The
  Earnings Announcement Premium and Trading Volume* (NBER WP 13090) — individual stocks earn
  higher average returns in the *month they announce*. Again a stock-level, announcement-timed
  premium that aggregates away at the index level when not surprise-conditioned.
- **Aggregate earnings and market returns.** Kothari, Lewellen & Warner (2006), *Stock Returns,
  Aggregate Earnings Surprises, and Behavioral Finance* (Journal of Financial Economics) —
  finds aggregate earnings news is, if anything, *negatively* related to contemporaneous market
  returns, the opposite of a naive "earnings season lifts the market" story.

## Calendar / seasonality anomalies — the methodological cousins

- **Calendar-effect skepticism.** Sullivan, Timmermann & White (2001), *Dangers of Data Mining:
  The Case of Calendar Effects in Stock Returns* (Journal of Econometrics) — once you correct
  for the many calendar windows that could have been tested, most apparent seasonals vanish.
  Directly relevant: four pre-chosen windows is already a four-way selection, and picking the
  strongest (October here) inflates significance.
- **The turn-of-the-month and holiday effects.** Lakonishok & Smidt (1988), *Are Seasonal
  Anomalies Real? A Ninety-Year Perspective* (Review of Financial Studies) — the template for
  honest in-window vs out-of-window calendar contrasts, and a caution on data-snooping.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) — the
  in-window dummy regression in [`strategy._hac_diff_tstat`](../earnings_season_tide/strategy.py).
- **Block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap* (JASA); Künsch
  (1989), *The Jackknife and the Bootstrap for General Stationary Observations* (Annals of
  Statistics) — the circular-block resampling behind the CI on the difference.
- **Excess-vs-excess Sharpe race.** The desk convention (see METHODOLOGY.md): a part-time-in-cash
  overlay is compared to buy-and-hold *after* both are reduced to excess of cash, so the
  overlay's idle cash does not flatter its Sharpe.

## Data sources used here

- **SPY total-return daily bars** (the desk's shared `_cache/SPY_total_return.parquet`,
  adjusted close carrying dividends — labelled total-return, not price-only), 1993–2025, with
  the in-progress year dropped. Headline numbers are pinned with an as-of date and content
  fingerprint (see [`results.md`](results.md)). The offline reproducible core and the
  test-suite run on the deterministic [`data.synthetic_daily`](../earnings_season_tide/data.py)
  generator and never touch the network.

## Related desk studies

- **[Study 34 — Aftershock](../../34-aftershock/)**: PEAD on a liquid single-stock universe —
  the *cross-sectional* effect this study's folk belief over-generalises to the index.
- **[Study 228 — Pre-Earnings Runup](../../228-pre-earnings-runup/)**: the *single-stock*
  pre-announcement drift; this study asks the orthogonal, index-level calendar question.
- **[Study 223 — Same-Month Seasonality](../../223-same-month-seasonality/)**: the desk's
  in-window calendar-contrast template (the one calendar seasonal that *did* clear the bar).
- **[Study 04 — Sell-in-May](../../04-sell-in-may/)** and the turn-of-month / pre-holiday
  studies: the broader family of market-wide calendar tides, mostly Mirage once charged costs.
