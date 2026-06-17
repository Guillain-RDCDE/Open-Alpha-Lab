# References & literature map — Study 284 (Equinox-Effect)

## The claim under test

The folk belief that the **equinoxes and solstices** — the four astronomical
"turning points" of the year — are also **market turning points** is a recurring
theme in market lore. It blends two strands: W. D. Gann-style "natural law" market
astrology (tops and bottoms supposedly cluster on seasonal cardinal dates) and the
softer seasonal-affective folk wisdom that the changing light of the year shifts
sentiment. The hypothesis we test is concrete: **the S&P 500 (^GSPC) earns an
abnormal return on the trading day of an equinox or solstice**, and/or some season
carries a distinct, significant pivot. There is no academic mechanism; this is a
celestial-calendar/sentiment claim, tested with an honest event study.

## Astronomy & seasonality background

- **Meeus, J. (1998).** *Astronomical Algorithms* (2nd ed.). Willmann-Bell.
  Chapter 27 ("Equinoxes and Solstices") gives the mean-instant polynomial plus
  the 24-term periodic correction we use to compute every equinox/solstice
  1928–2026. Our hardcoded `EQUINOX_SOLSTICE` table is generated from it and
  cross-checked to the minute against published USNO / timeanddate.com tables.
- **United States Naval Observatory, "Earth's Seasons."** The authoritative public
  table of equinox/solstice instants (UT). Used to verify the Meeus dates
  (e.g. 2024: Mar 20, Jun 20, Sep 22, Dec 21 — all exact).
- **Kamstra, M. J., Kramer, L. A. & Levi, M. D. (2003).** "Winter Blues: A SAD
  Stock Market Cycle." *American Economic Review*, 93(1), 324–343. The most serious
  "length-of-day affects returns" claim — it argues seasonal affective disorder
  drives a fall-to-winter return pattern tied to the *solstice*. Later work
  (Kelly & Meschke 2010) attributes much of it to the well-known turn-of-year and
  September/October seasonals — directly relevant: any "equinox/solstice" effect we
  find is most plausibly those calendar seasonals, not the astronomy.

## Why this is a textbook mirage

- **No mechanism.** The Earth's axial tilt changing sign does not change earnings,
  rates, or liquidity on any one day. Any "effect" must be pure sentiment — and a
  single-day sentiment effect on a fixed astronomical date would be trivially
  arbitraged.
- **Seasonal confound.** There *are* real calendar effects (Sell-in-May, the
  Santa-Claus rally, September weakness). The autumn equinox sits in weak
  September; the winter solstice sits inside the year-end rally window. An "equinox
  effect" is mostly a noisy re-slice of those — see the autumn result in
  `docs/results.md`.
- **Multiple comparisons.** Slicing by kind (equinox/solstice) and by the four
  seasons manufactures lucky sub-samples. We Bonferroni-correct over 7 slices and
  show the one mildly-interesting slice (autumn) evaporates — and it never cleared
  0.05 raw anyway.
- **The base rate.** The market drifts up ~3 bps/day regardless; abnormal returns
  must be measured against that drift, not against zero.

## Method lineage

- **Event study / CAR.** Classic MacKinlay (1997, *JEL*) market-model event study,
  simplified to a flat (constant-mean) model because the four annual instants are
  too regular for a rolling estimation window to add anything. AAR = cross-event
  mean abnormal return per relative day; CAR = its cumulative sum over the window.
- **Newey-West (HAC) t-stat.** Heteroskedasticity- and autocorrelation-consistent
  standard error (Newey & West 1987) on the event-day abnormal returns, with a few
  lags to absorb the mild dependence when nearby event windows overlap.
- **Permutation test.** Draw `n_events` random trading days 10,000 times, record
  the distribution of mean abnormal returns, and read off the two-sided p-value —
  the assumption-light null for a regular but small event set.
- **Bonferroni correction.** Multiply the per-slice p-value by the number of slices
  tested (7) — a deliberately conservative guard against slice-mining.

## Data sources

- **^GSPC daily price series.** Split-adjusted closes staged at
  `_cache/^GSPC_split_only.parquet` (1928–2026), read-only. Price-only (no
  dividends); the daily mean drift is used as the abnormal-return baseline.
- **Equinox/solstice table.** Hardcoded in `data.py`, computed from Meeus (1998)
  ch. 27 and verified against the USNO "Earth's Seasons" table.

## Related desk studies

- **[Study 280 — Solar-Eclipse](../../280-solar-eclipse/)**: the other astronomical
  "omen" event study — eclipses vs the S&P, same None/Mirage family and the same
  ^GSPC-event template.
- **[Study 158 — Super-Bowl](../../158-super-bowl/)**: the hardcoded-event-table +
  real-market-returns pattern this study mirrors.
- **SAD / change-of-clock / lunar seasonality** studies in the calendar-quirk
  family (Lot 138-157) — same "length-of-day / mood moves markets" hypothesis class.
