# References & literature map — Study 164 (Mercury-Retrograde)

## The claim under test

- **Financial astrology's signature warning.** Practitioners of financial astrology and many
  retail investors claim that Mercury retrograde — when the planet appears to move backwards
  from Earth's perspective (an optical illusion due to orbital mechanics) — causes market
  chaos, communication failures, poor investment decisions, and generally bearish returns.
  The period occurs roughly three times per year for about three weeks each time (~19% of
  calendar days). The claim appears across astrology-finance websites, in media pieces
  (e.g., Business Insider, MarketWatch), and is referenced in surveys of individual-investor
  sentiment. We steelman it as: *"S&P 500 daily returns are significantly lower, and/or
  significantly more volatile, on Mercury-retrograde trading days compared to non-retrograde
  days, in a way that survives a permutation control and is consistent across market regimes."*

## Academic literature on financial astrology and lunar/planetary effects

- **Dichev & Janes (2003).** *Lunar Cycle Effects in Stock Returns.* Journal of Private
  Equity. Documents higher returns in the two-week period around the new moon than around
  the full moon across global stock markets. The most-cited "lunar" finance paper. Our study
  tests a different planet but applies the same discipline: a robust statistical comparison
  against a matched control.
- **Hirshleifer & Shumway (2003).** *Good Day Sunshine: Stock Returns and the Weather.*
  Journal of Finance, 58(3), 1009–1032. Sunny weather correlates with positive stock
  returns in 26 countries. A genuine anomaly driven by investor mood — the closest analogue
  to the astrological-mood channel. The difference: weather is local, measurable, and
  psychologically plausible; Mercury's apparent motion is not.
- **Cao & Wei (2005).** *Stock Market Returns: A Note on Temperature Anomaly.*
  Journal of Banking & Finance, 29(6), 1559–1573. Temperature predicts stock returns —
  another mood-based anomaly suggesting investor psychology is real, but does not extend
  to planetary mechanics.
- **Bergsma & Jiang (2016).** *Cultural New Year Celebrations and Stock Returns.*
  Journal of Empirical Finance. Documents cultural-calendar effects that are persistent and
  arguably behavioural. A legitimate calendar anomaly — unlike Mercury retrograde.
- **Lucey & Dowling (2005).** *The Role of Feelings in Investor Decision-Making.*
  Journal of Economic Surveys, 19(2), 211–237. Review of mood effects in finance; covers
  weather, biorhythm, and sporting outcomes. No evidence for planetary effects.
- **Rotton & Kelly (1985).** *Much Ado About the Full Moon: A Meta-Analysis of Lunar-Lunacy
  Research.* Psychological Bulletin, 97(2), 286–306. A careful meta-analysis finding no
  reliable relationship between lunar cycle and human behaviour — the progenitor of the
  discipline we apply to Mercury retrograde.

## The statistical issues specific to this claim

- **Small-sample / coincidence problem.** Mercury retrograde occurs ~3 times per year,
  so the 2000–2026 window contains only ~75 retrograde periods. Each period is approximately
  three weeks. Major market crises (dot-com 2000–2002, GFC 2008, COVID 2020) each cover
  weeks to months, and by pure chance some coincide with retrograde windows. With n=75
  periods the claim cannot be distinguished from random coincidence. See Sullivan, Timmermann
  & White (1999), *Data-Snooping, Technical Trading Rule Performance, and the Bootstrap*
  (Journal of Finance) for the general framework.
- **Multiple comparison and data mining.** Testing the same retrograde claim on multiple
  measures (mean return, volatility, strategy CAGR) inflates the family-wise type I error.
  A Bonferroni correction for three tests would require |t| ≥ 2.39 — our Welch *t* of
  −2.21 does not clear this bar. See Harvey, Liu & Zhu (2016), *…and the Cross-Section
  of Expected Returns* (Review of Financial Studies), for the broader multiple-testing
  problem in factor research.
- **Regime coincidence, not causation.** The 2010–2019 subperiod, the most stable recent
  market decade, shows Welch *t* = −0.11 for the retrograde vs direct comparison. The
  pooled −2.21 is driven by the dot-com/GFC/COVID episodes — crises that happened to
  coincide with some retrograde windows by chance. See Fama (1991), *Efficient Capital
  Markets: II* (Journal of Finance) for the general argument that anomalies that disappear
  in calm markets are regime-coincidences, not structural signals.

## The null expected from astronomy / market microstructure

- **Fama (1970).** *Efficient Capital Markets: A Review of Theory and Empirical Work.*
  Journal of Finance. Daily returns are close to a random walk in the weak-form sense.
  For Mercury retrograde to matter, it must enter the information set of market participants
  — either through direct astrology-based trading or through a psychological channel. The
  former requires a large coordinated group; the latter requires planetary motion to reliably
  affect mood more than any of hundreds of correlated daily events.
- **Roll (1984).** *A Simple Implicit Measure of the Effective Bid-Ask Spread.* Journal of
  Finance. Microstructure noise at daily frequency swamps any plausible tiny planetary
  mood signal.

## The ephemeris data source

The retrograde period table hardcoded in `data.py` is cross-checked against:

- **NASA JPL Horizons ephemeris** (ssd.jpl.nasa.gov) — the authoritative source for
  planetary positions, used by professional astronomers and space agencies.
- **TimeandDate.com/astronomy** — cross-reference for Mercury retrograde dates, widely
  used by astrology-finance practitioners as the "official" retrograde calendar.
- **Astrology.com retrograde calendar** — the retail practitioner's standard source,
  which our dates match within ±1 day accounting for time-zone conventions.

The dates are deterministic astronomical facts and are fully reproducible from any
ephemeris library (e.g., `ephem`, `astropy`, `skyfield`).

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy._hac_tstat`](../mercury_retrograde/strategy.py) implements this for both
  the retrograde-only mean and the full-period baseline.
- **Levene variance test.** Levene (1960), *Robust Tests for Equality of Variances*,
  from *Contributions to Probability and Statistics* (Olkin, ed.) — used for the
  retrograde vs direct volatility comparison.
- **Permutation / randomisation test.** Good (2004), *Permutation, Parametric and Bootstrap
  Tests of Hypotheses* (Springer) — the null baseline in `strategy.permutation_null`.
- **Reproducibility stamp.** `data.fingerprint` pins the headline run to a content hash
  of the ^GSPC close-price series.

## Related desk studies

- **[Study 136 — Mark-Twain](../../136-mark-twain/)**: October-effect teardown — same
  family of calendar-myth busting, same machinery, same honest conclusion.
- **[Study 48 — Groundhog](../../48-groundhog/)**: another weather-based market forecast
  subjected to the same treatment.
- **[Study 80 — Cold-Open](../../80-cold-open/)**: January Barometer — the classic
  calendar anomaly with the largest real-world following.
- **[Study 83 — Half-Life](../../83-half-life/)**: an n=tiny teardown showing how small
  samples produce large t-statistics that mean nothing.
- **[Study 76 — Rice-Paper](../../76-rice-paper/)**: Bonferroni correction applied to a
  multi-signal study — the multiple-comparisons discipline this study invokes.
