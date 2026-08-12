# References & literature map — Study 879 (Weekly Economic Index)

## The claim under test

- **The source paper.** Daniel J. **Lewis, Karel Mertens & James H. Stock**, *"U.S.
  Economic Activity during the Early Weeks of the SARS-CoV-2 Outbreak"* (2020) and the
  companion *"Monitoring Economic Activity in Real Time Using the Weekly Economic Index"*
  (Liberty Street Economics / NBER Working Paper 26937). They combine **ten weekly**
  activity series into a single index that tracks the year-over-year growth rate of a broad
  set of monthly and quarterly activity measures — a genuine **high-frequency nowcast** of
  real activity, updated every week rather than monthly.
- **The ten inputs.** Redbook same-store retail sales; initial and continuing
  unemployment-insurance claims; adjusted federal income-tax withholding; railroad traffic
  (American Association of Railroads); retail fuel sales; the American Staffing Association
  temporary-staffing index; steel production; electricity output; and Rasmussen consumer
  confidence. The series are standardized, expressed in annual log changes, and combined
  with the published weights (a factor-model projection onto the monthly activity space).
- **Where it lives now.** Originally published by the **New York Fed**, the WEI moved to
  the **Federal Reserve Bank of Dallas** (authors Tyler Atkinson, Isaiah Spellman et al.),
  which posts the full weekly history and the current weights in a public workbook. It is
  also mirrored on **FRED** as series `WEI`.
- **The specific test here.** Does the WEI's *level* and *weekly change* predict **forward
  SPY** returns and the **cyclical-vs-defensive rotation** (consumer-discretionary `XLY`
  minus consumer-staples `XLP`)? We run a weekly predictive regression with Newey-West HAC
  *t*, a two-era robustness cut, a permutation placebo, a costed rotation overlay, and a
  seeded synthetic positive control. The economic question: does *higher-frequency* growth
  information beat what the monthly macro tape already tells the market, or is the nowcast a
  smooth proxy for the recession/recovery cycle whose apparent edge is just trend?

## What we measure, and the honesty rails

- **The nowcast, straight from the source.** The current-vintage weekly WEI level and its
  week-over-week change, taken from the Dallas Fed workbook (no reconstruction).
- **Point-in-time, one documented lag.** A week-ending-Saturday WEI is only *published* the
  following week, so every forward market return is anchored **one trading week (5 days)
  later** — the position is taken after the nowcast is known. Zero look-ahead in the timing;
  the level result carries a documented **revision** caveat (we use the revised vintage),
  named on the Signal axis.
- **Robust inference.** A Newey-West (HAC, Bartlett, 8-lag) *t* on the predictive-regression
  slopes — overlapping weekly forward windows are serially correlated, so a plain OLS *t*
  would overstate significance. Univariate and joint slopes; a two-era cut (2008–2016 vs
  2017–2026); a 2,000-draw permutation placebo that breaks the signal→return link.
- **The timer is graded separately.** The rotation overlay charges one-way cost per leg on
  turnover and borrow on the short — the honest test of whether a thin weekly edge survives
  friction.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the regression slopes and the spread series).
- **Wilson, E. B. (1927)** — score interval for a binomial share (win-rate bands).
- **Welch, B. L. (1947)** — the unequal-variance *t* used for the conditional-vs-base test.

## Data sources

- **Dallas Fed WEI workbook** — `https://www.dallasfed.org/-/media/documents/research/wei/weekly-economic-index.xlsx`,
  `2008-current` sheet, weekly week-ending Saturdays 2008-01-05 → 2026-06-27. Documented
  fallback: **FRED** series `WEI` (`https://fred.stlouisfed.org/graph/fredgraph.csv?id=WEI`).
- **yfinance daily total-return closes** for SPY, XLY, XLP through 2026-06-30, cached under
  `_cache/`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [384-ism-pmi-regime](../../384-ism-pmi-regime/) — the **monthly** ISM/PMI diffusion index
  as a regime switch. This study uses a **weekly** blended nowcast (10 high-frequency
  series), and asks specifically whether the *higher frequency* adds timing power.
- [387-economic-surprise-index](../../387-economic-surprise-index/) — a **surprise** index
  (actual minus a trailing-consensus proxy) on monthly data. The WEI is a **level nowcast**,
  not a surprise-vs-expectation gap.
- [626-unemployment-trend-timing](../../626-unemployment-trend-timing/) — the **monthly
  unemployment-rate trend** (a single labour series). The WEI's claims are two of its ten
  inputs, but this study tests the **composite weekly index**, not the labour trend alone.
- [757-cass-freight](../../757-cass-freight/) — the **monthly** Cass Freight shipments index
  (a single transport series). This study is the **weekly, ten-series** composite and adds a
  **cyclical-vs-defensive rotation** target (XLY−XLP) absent there.

None of the siblings tests the **weekly, ten-series WEI level & change against forward SPY
and the XLY−XLP rotation** — this study's own axis.
