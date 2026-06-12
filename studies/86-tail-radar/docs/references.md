# References & literature map — Study 86 (Tail-Radar)

## The claim under test

- **The CBOE SKEW index as a black-swan radar.** The CBOE publishes the SKEW index as a
  measure of the perceived probability of outlier returns in the S&P 500 (formally: the
  risk-neutral skewness of the 30-day return distribution, extracted from OTM put option
  prices). The folk claim is that a high SKEW reading signals that "smart money" is hedging
  against a coming crash, and that investors should therefore reduce equity exposure or buy
  protection when SKEW is elevated. We steelman this as: *the SKEW quintile spread
  (Q5 − Q1 forward SPY returns) is negative and statistically significant, and the
  frequency of large forward losses (> −5%) is elevated after high-SKEW readings.* The
  claim originates with CBOE's own SKEW white papers and is widely repeated in financial
  media (CNBC, Bloomberg, financial advisor commentary circa 2010–2020).

## The theoretical foundation — why SKEW *could* matter

- **Risk-neutral skewness and crash risk.** Bakshi, Kapadia & Madan (2003), *Stock Return
  Characteristics, Skew Laws, and the Differential Pricing of Individual Equity Options*
  (Review of Financial Studies) — the theoretical framework for extracting higher moments
  from option prices. Elevated put-skew should, in theory, reflect elevated crash risk
  priced by informed hedgers.
- **Fear gauge evidence from VIX.** Whaley (2000), *The Investor Fear Gauge* (Journal of
  Portfolio Management) — VIX as a market thermometer. The SKEW claim is an extension:
  if VIX measures the *level* of fear, SKEW measures the *shape* (left-tail loading).
  Empirically, VIX is a much stronger predictor of realised volatility than SKEW.
- **Informed options trading around crashes.** Pan & Poteshman (2006), *The Information in
  Option Volume for Future Stock Prices* (Review of Financial Studies) — some options
  activity is informationally motivated. The SKEW claim requires this to aggregate to the
  index level and to be forward-looking at a tradable horizon; our results show it does not.

## Why the claim is likely to fail — the sceptical literature

- **SKEW does not forecast returns.** Stilger, Kostakis & Poon (2017), *What Does Risk-
  Neutral Skewness Tell Us About Future Stock Returns?* (Management Science) — individual
  stock risk-neutral skewness has mild cross-sectional predictive power but the evidence
  for index-level SKEW predicting aggregate market returns is weak. Our study is the index-
  level test.
- **SKEW as insurance premium, not oracle.** The CBOE itself notes that SKEW is *not* a
  crash predictor but a measure of *priced tail risk* — options may be expensive because
  fear is high, even when the fear is irrational. See CBOE (2011), *CBOE SKEW Index White
  Paper* (cboe.com). High SKEW is consistent with high demand for crash protection,
  which can persist for years without a crash ever materialising.
- **Volatility risk premium and the wrong direction.** Carr & Wu (2011), *Variance Risk
  Premiums* (Review of Financial Studies) — the volatility risk premium means put options
  are systematically overpriced. High SKEW (expensive puts) can co-occur with *rising*
  future markets because the fear premium dissipates. This matches our Q1-beats-Q5 finding.
- **Data-snooping risk in volatility indices.** Harvey, Liu & Zhu (2016), *… and the
  Cross-Section of Expected Returns* (Review of Financial Studies) — with many fear
  proxies tested, some will show spurious predictive power; the SKEW claim is driven
  partly by data-mining on short samples.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy._hac_tstat`](../tail_radar/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Fisher exact test for 2×2 frequency tables.** Fisher (1922) — the crash-frequency
  comparison; implemented via `scipy.stats.fisher_exact`.
- **Rolling out-of-sample percentile ranking.** Signal construction uses only history
  available at date *t* — strictly no look-ahead. Rolling 252-day window to adapt to
  the secular drift of SKEW (it trended up from ~120 in the 1990s to ~130+ post-GFC).
- **Reproducibility stamp.** [`quantlab/repro.py`](../../../quantlab/repro.py) and
  [`data.fingerprint`](../tail_radar/data.py) — the as-of freeze and content fingerprint.

## Data sources used here

- **Yahoo Finance daily closes** (via `yfinance`): `^SKEW`, `^VIX`, `SPY` from 1993 to
  present. SKEW history begins in 1990; SPY begins in 1993; the overlap is 1993-02-01.
  All three series are cached as a single parquet under `_cache/`. The offline
  reproducible core and test-suite run on the deterministic
  [`data.synthetic_daily`](../tail_radar/data.py) generator, never the network.

## Related desk studies

- **[Study 42 — Last-Call](../../42-last-call/)**: VIX term-structure slope as a
  risk-on signal — the closest relative; VIX structure carries more signal than level.
- **[Study 70 — Digital-Gold](../../70-digital-gold/)**: crypto tail-risk — another
  "does this fear measure predict crashes?" study.
- **[Study 48 — Groundhog](../../48-groundhog/)**: calendar effects and return patterns —
  demonstrates how low-SKEW / low-VIX environments (risk-on) are associated with the
  strongest calendar-return windows.
- **[Study 67 — Fed-Drift](../../67-fed-drift/)**: event windows around FOMC dates —
  another example where a widely cited predictive signal (here macro, there options-based)
  is tested honestly and found to be weak or absent at tradable horizons.
