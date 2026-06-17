# References & literature map — Study 280 (Solar-Eclipse)

## The claim under test

The folk belief that solar eclipses are bad omens — and therefore "spook" the
stock market — is ancient and unkillable. The hypothesis we test is concrete:
**the S&P 500 (^GSPC) earns an abnormally negative return on the trading day of a
solar eclipse**, especially for the rare eclipses whose path of totality crosses
the United States. There is no academic mechanism; this is a sentiment/astrology
claim, tested with an honest event study.

## Astronomy & sentiment background

- **Espenak, F. & Meeus, J. (2006).** *Five Millennium Canon of Solar Eclipses:
  -1999 to +3000.* NASA/TP-2006-214141. The authoritative catalog of every solar
  eclipse with date, type (total/annular/hybrid) and central path. Our hardcoded
  `SOLAR_ECLIPSES` table (1930–2026) and the `us_path` flag are sourced here and
  cross-checked against Wikipedia's per-century lists.
- **Krivelyova, A. & Robotti, C. (2003).** "Playing the Field: Geomagnetic Storms
  and the Stock Market." *Federal Reserve Bank of Atlanta WP 2003-5.* The closest
  serious cousin: it argues *geomagnetic* activity (not eclipses) correlates with
  returns via mood. Widely cited, widely doubted — a reminder that "celestial
  sentiment" effects are fragile and rarely replicate out of sample.
- **Yuan, K., Zheng, L. & Zhu, Q. (2006).** "Are Investors Moonstruck? Lunar
  Phases and Stock Returns." *Journal of Empirical Finance*, 13(1), 1–23. Reports
  a new-moon vs full-moon return gap; later work (Herbst 2007) attributes most of
  it to data-mining and overlapping windows. Eclipses are special syzygies, so
  this is the most directly relevant "celestial calendar" anomaly literature.

## Why this is a textbook mirage

- **No mechanism.** A solar eclipse changes daylight for minutes over a narrow
  path; it does not change earnings, rates, or liquidity. Any "effect" must be
  pure sentiment — and sentiment effects this specific rarely survive honest tests.
- **Tiny n.** ~80 central eclipses in the ^GSPC era, only ~12 crossing the US.
  With ~1% daily vol, the event-day mean SE is ~11 bps, so effects below ~25 bps
  are undetectable. See the power note in `docs/results.md`.
- **Multiple comparisons.** Slicing by type (total/annular/hybrid) and geography
  (US-path) manufactures lucky sub-samples. We Bonferroni-correct and show the
  one "significant" slice (annular) evaporates.
- **The base rate.** The market drifts up ~3 bps/day regardless; abnormal returns
  must be measured against that drift, not against zero — otherwise every up-day
  bias is mistaken for signal.

## Method lineage

- **Event study / CAR.** Classic MacKinlay (1997, *JEL*) market-model event study,
  simplified to a flat (constant-mean) model because eclipses are too sparse for a
  rolling estimation window to add anything. AAR = cross-event mean abnormal
  return per relative day; CAR = its cumulative sum over the window.
- **Newey-West (HAC) t-stat.** Heteroskedasticity- and autocorrelation-consistent
  standard error (Newey & West 1987) on the event-day abnormal returns, with a few
  lags to absorb the mild dependence when nearby eclipse windows overlap.
- **Permutation test.** Draw `n_eclipses` random trading days 10,000 times, record
  the distribution of mean abnormal returns, and read off the two-sided p-value —
  the assumption-light null for a small, irregular event set.
- **Bonferroni correction.** Multiply the per-slice p-value by the number of
  slices tested (4) — a deliberately conservative guard against slice-mining.

## Data sources

- **^GSPC daily price series.** Split-adjusted closes staged at
  `_cache/^GSPC_split_only.parquet` (1928–2026), read-only. Price-only (no
  dividends); the daily mean drift is used as the abnormal-return baseline.
- **Solar-eclipse table.** Hardcoded in `data.py`. Source: NASA Five Millennium
  Canon (Espenak & Meeus) + Wikipedia "List of solar eclipses in the 20th/21st
  century."

## Related desk studies

- **[Study 164 — Mercury-Retrograde](../../164-mercury-retrograde/)**: the other
  "celestial omen" teardown — astrology vs the S&P, same None/Mirage family.
- **[Study 158 — Super-Bowl](../../158-super-bowl/)**: the hardcoded-event-table +
  real-market-returns pattern this study mirrors.
- **Lunar / SAD / change-of-clock seasonality** studies in the calendar-quirk
  family (Lot 138-157) — same "mood moves markets" hypothesis class.
