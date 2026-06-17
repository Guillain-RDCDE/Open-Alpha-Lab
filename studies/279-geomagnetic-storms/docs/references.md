# References & literature map — Study 279 (Geomagnetic-Storms)

## The claim under test

**Krivelyova, A. & Robotti, C. (2003).** "Playing the Field: Geomagnetic Storms and
International Stock Markets." *Federal Reserve Bank of Atlanta Working Paper 2003-5.*
The canonical study. The authors match daily geomagnetic-activity indices (the planetary
Kp/Ap index) to stock returns across the US and other markets and find that returns in the
days/weeks **following** geomagnetically stormy periods are economically and (in their tests)
statistically lower than returns following calm periods. The proposed channel is
mood-misattribution: geomagnetic storms are associated with elevated clinical depression and
anxiety, and a gloomier marginal investor demands a higher risk premium (lower prices now,
lower realised returns through the storm window).

## The mood-misattribution channel

- **Saunders, E. M. (1993).** "Stock Prices and Wall Street Weather." *American Economic
  Review*, 83(5), 1337–1345. The founding "mood affects prices" paper (NYC cloud cover vs
  returns). Geomagnetic storms are the same family with an *exogenous, non-local* driver.
- **Hirshleifer, D. & Shumway, T. (2003).** "Good Day Sunshine: Stock Returns and the
  Weather." *Journal of Finance*, 58(3), 1009–1032. Sunshine at the exchange city predicts
  higher returns; establishes the misattribution mechanism (people misread mood as information).
- **Kamstra, M., Kramer, L. & Levi, M. (2003).** "Winter Blues: A SAD Stock Market Cycle."
  *American Economic Review*, 93(1), 324–343. Seasonal-affective disorder and the length of
  night drive a seasonal return pattern — the same mood logic on an annual cycle.
- **Dowling, M. & Lucey, B. (2005).** "Weather, Biorhythms, Beliefs and Stock Returns."
  *International Review of Financial Analysis*, 14(3), 337–355. Includes geomagnetic storms
  among the "biorhythm" proxies and broadly corroborates the Krivelyova-Robotti direction.

## Why this is a genuine REAL *candidate* (and why it still lands at WEAK here)

- **Exogeneity.** Unlike accounting or price-based factors, the geomagnetic index is a
  physical measurement no firm manages and no arbitrageur can exhaust — there is no obvious
  reverse-causality or data-snooping path from returns to the Sun. That makes the *direction*
  credible.
- **The honest standard error.** Both geomagnetic activity and equity returns cluster in time
  (storms arrive in multi-month bursts during the solar cycle's declining phase; returns have
  mild momentum/serial correlation). A naive i.i.d. t-test overstates significance. We use a
  **Newey-West HAC** t-stat (auto bandwidth, Newey-West 1994), which is the appropriate, more
  conservative inference. On the real tape it gives **t = 1.88** — under the desk's |t| ≥ 2 bar.
- **Power and thresholds.** With ~226 stormy and ~226 calm months and ~4.3%/month volatility,
  a ~0.8%/month gap is right at the edge of detectability. Defining storm/calm in-sample
  (whole-sample quantiles) mildly flatters the gap; a walk-forward threshold weakens it.

## The verdict logic (desk rubric)

- **Signal = WEAK.** Right sign, economically large, mechanism-backed, perm p = 0.046 — but
  the robust HAC t-stat is 1.88, below 2.0, and below 2.0 in every sub-period. The rubric:
  *REAL requires a robust HAC |t| ≥ 2 on the real tape; literature support alone is WEAK.*
- **Tradability = MIRAGE.** The lagged, costed long-calm / short-storm overlay earns
  +2.4%/yr net vs +8.7%/yr buy-and-hold; the strategy sits flat ~60% of months and pays borrow
  while short. There is no edge to harvest.

## Related desk studies (mood / calendar / exogenous-driver anomalies)

- **Sunshine / weather** — the local-mood analogue of the storm channel.
- **Lunar-cycle and SAD studies** — mood on monthly and annual cycles.
- **[Study 158 — Super-Bowl](../../158-super-bowl/)** — a *folklore* contrast: no mechanism,
  no signal (None / Mirage). Geomagnetic storms are the opposite end of the spectrum: a real
  mechanism and the right sign, falling just short of significance.

## Method lineage

- **Newey, W. K. & West, K. D. (1987, 1994).** "A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix" and the 1994 automatic
  bandwidth selection. The HAC t-stat used throughout `strategy.py`.
- **Permutation test.** Shuffle calm/stormy labels 10,000 times; the p-value is the fraction
  of shuffles whose gap equals or exceeds the observed gap. A complementary, distribution-free
  check that nonetheless assumes month exchangeability (hence over-credits a clustered effect).
- **Welch t-test.** `scipy.stats.ttest_ind(equal_var=False)` as an independence-assuming
  cross-check on the two-sample mean difference.

## Data sources

- **^GSPC daily close.** Yahoo Finance via the repo-level cache
  `_cache/^GSPC_split_only.parquet`, resampled to month-end. **Price index, no dividends** —
  all real-tape returns are price-only and labelled as such.
- **Geomagnetic Ap index.** Reconstructed deterministically in `data.py` from the canonical
  solar-cycle minima (cycles 17–25, SILSO / NOAA SWPC) plus documented great-storm spikes
  (Aug 1972, Mar 1989, Oct/Nov 2003, Mar 2015, May 2024). The reconstruction encodes the
  declining-phase geomagnetic peak and the semi-annual (equinoctial / Russell-McPherron)
  modulation. It is a pure function of the calendar — byte-for-byte reproducible offline. A
  reader wanting the official daily Ap series can pull it from GFZ Potsdam / NOAA and re-bin;
  the monthly regime classification is robust to that substitution.
