# References & literature map — Study 148 (Lunar-Effect)

## The claim under test

- **Yuan, Zheng & Zhu (2006).** "Are Investors Moonstruck? Lunar Phases and Stock Returns,"
  *Journal of Empirical Finance* 13(1), 1–23. The original paper: using 48 countries over
  1973–2001, the authors report mean daily stock returns around new moons are ~3–5 bps
  higher than around full moons, attributing the gap to "investor mood" driven by
  disrupted sleep near full moons.  This is the study's canonical source and the claim we
  test.  They find the global composite effect is significant, but note it is absent in
  many individual markets.

## Why the steelman is almost coherent — the mood-and-sleep channel

- **Cajochen et al. (2013).** "Evidence that the Lunar Cycle Influences Human Sleep,"
  *Current Biology* 23(15), 1485–1488. Documents disrupted sleep near full moons in a
  controlled study, providing a plausible biological mechanism for the YZZ claim:
  worse sleep → worse mood → more risk-averse decisions → lower equity demand.
- **Hirshleifer & Shumway (2003).** "Good Day Sunshine: Stock Returns and the Weather,"
  *Journal of Finance* 58(3), 1009–1032. Documents that sunshine correlates with daily
  returns — the broader literature on environmental mood-effects on markets that gives
  the lunar claim a family to sit in.  Sunshine is at least a contemporaneous
  environmental signal; the lunar claim is structurally the same but far weaker.
- **Dichev & Janes (2003).** "Lunar Cycle Effects in Stock Returns," *Journal of Private
  Equity* 6(4), 8–29. An earlier paper reporting similar lunar patterns in US data
  1973–2001 (US, UK), predating YZZ.  We treat YZZ as the definitive reference because
  it used the broadest cross-country sample.

## Why it likely fails — and how it fails here

- **Lunar phase is random noise with respect to returns.**  The moon's 29.53-day cycle
  is incommensurable with the trading calendar and uncorrelated with any known
  macroeconomic or earnings cycle.  Once the alleged channel (sleep → mood → trading)
  is subject to a fully powered test — nearly 100 years, 24,727 trading days — the
  contrast (NEW − FULL = +1.06 bps/day, HAC *t* = 0.75) is statistically noise.
- **McLean & Pontiff (2016).** "Does Publishing Research Weaken the Anomaly?"
  *Journal of Finance* 71(1), 5–46. Documents that post-publication returns on
  finance anomalies decay on average ~26%.  The lunar effect was already marginal
  and never close to actionable — it has not survived out-of-sample.
- **Lo & MacKinlay (1990).** "Data-Snooping Biases in Tests of Financial Asset
  Pricing Models," *Review of Financial Studies* 3(3), 431–467.  The lunar calendar
  is one of thousands of potential partitions of a return series; finding a single
  significant partition in a search across many is expected by chance alone.  YZZ's
  multi-country panel offers some protection but the effect disappears in a fully
  powered single-market long-run test.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), "A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix," *Econometrica*
  55(3), 703–708 — [`strategy.hac_tstat`](../lunar_effect/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Block bootstrap CI.** Politis & Romano (1994), "The Stationary Bootstrap," *JASA*
  89(428), 1303–1313 — [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).
- **Permutation / shuffled-label control.** Destroys any genuine lunar association
  while preserving the return distribution: [`strategy.shuffled_contrast`](../lunar_effect/strategy.py).
- **Lunar phase computation.** Meeus, J. (1998), *Astronomical Algorithms*, 2nd ed.,
  Willmann-Bell.  Mean synodic month 29.530588853 days; J2000.0 new-moon epoch JD
  2451551.26 (6 January 2000 18:14 UTC).  Mean-moon formula; actual new/full moons
  deviate up to ~1 day due to orbital eccentricity, but the ±7-day window absorbs this.

## Data sources used here

- **S&P 500 (^GSPC) daily returns, 1928-01-03 to 2026-06-11** — shared repo cache at
  `_cache/last_call_gspc.parquet` (sourced via `yfinance`).  Fingerprint: `b9c7f7255063`.
  No external data fetch is required for the lunar phase — pure astronomy.
- **Lunar phase engine** — entirely in-code (`data.lunar_phase`, `data.lunar_label`),
  using only numpy and pandas.  The offline test-suite and the synthetic positive control
  run without any network access or calendar lookup.

## Related desk studies

- **[Study 55 — Summer-Lull](../../55-summer-lull/)** and
  **[Study 48 — Groundhog](../../48-groundhog/)**: seasonal/calendar effects — the same
  family of "time-of-year anomaly" ideas; both evaporate on honest testing.
- **[Study 67 — Fed-Drift](../../67-fed-drift/)** and
  **[Study 135 — FOMC-Cycle](../../135-fomc-cycle/)**: event-driven calendar effects —
  genuinely periodic events that do carry measurable return signatures, in contrast to
  the purely astronomical lunar cycle.
- **[Study 82 — Witching-Hour](../../82-witching-hour/)**: another calendar-driven claim
  (expiry effects) — the desk's broadest event-calendar teardown.
