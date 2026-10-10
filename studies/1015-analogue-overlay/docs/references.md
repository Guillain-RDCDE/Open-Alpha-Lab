# Sources & literature map — Study 1015 (The 1929 Overlay)

## The claim

- **The "1929 overlay" chart.** A recurring piece of market folklore rather than a paper: today's
  index path laid over the run-up to the 1929 crash (or 1987, or 2000), rescaled until the lines
  coincide, often with a correlation printed in the corner. A version overlaying the 2012-14 Dow
  on 1928-29 circulated widely in early 2014; the market rose over the following year (section 3
  of [results.md](results.md) scores that kind of overlay on the monthly tape). No single author
  owns the idea, which is why the study tests the general forecaster rather than one chart.

## Analogue (nearest-neighbour) forecasting

- **Lorenz, E. N. (1969), "Atmospheric Predictability as Revealed by Naturally Occurring
  Analogues", *Journal of the Atmospheric Sciences* 26(4).** The method's origin: find past
  states that resemble today and read the future off what followed them — and the finding that
  good analogues are rare once the state is high-dimensional.
- **Farmer, J. D. & Sidorowich, J. J. (1987), "Predicting Chaotic Time Series", *Physical Review
  Letters* 59(8).** Nearest-neighbour prediction in reconstructed state space; the template for
  the "k analogues, average what followed" forecaster here.
- **Lo, A. W., Mamaysky, H. & Wang, J. (2000), "Foundations of Technical Analysis: Computational
  Algorithms, Statistical Inference, and Empirical Implementation", *Journal of Finance* 55(4).**
  The serious attempt to make visual chart patterns testable; the spirit of turning a picture
  into a forecaster with a null.

## Why trending paths correlate — the spurious-correlation core

- **Yule, G. U. (1926), "Why Do We Sometimes Get Nonsense-Correlations between Time-Series?",
  *Journal of the Royal Statistical Society* 89(1).** The original demonstration that two
  unrelated integrated series show large sample correlations — the overlay chart's whole trick.
- **Granger, C. W. J. & Newbold, P. (1974), "Spurious Regressions in Econometrics", *Journal of
  Econometrics* 2(2).** Regressing one random walk on another produces "significant" fits.
- **Phillips, P. C. B. (1986), "Understanding Spurious Regressions in Econometrics", *Journal of
  Econometrics* 33(3).** The asymptotic theory: the sample correlation of two independent random
  walks does not converge to zero but to a non-degenerate random variable.
- **Ernst, P. A., Shepp, L. A. & Wyner, A. J. (2017), "Yule's 'Nonsense Correlation' Solved!",
  *Annals of Statistics* 45(4).** The variance of that limiting correlation, in closed form —
  section 1 of the results is its Monte Carlo counterpart with drift.

## Inference and the search

- **Newey, W. K. & West, K. D. (1987), "A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix", *Econometrica* 55(3), 703-708.** Overlapping
  h-period outcomes.
- **Politis, D. N. & Romano, J. P. (1994), "The Stationary Bootstrap", *Journal of the American
  Statistical Association* 89(428).** Block-bootstrap intervals and the Reality Check resampling.
- **White, H. (2000), "A Reality Check for Data Snooping", *Econometrica* 68(5), 1097-1126.**
  The correction applied over every timing rule in the grid.
- **Sullivan, R., Timmermann, A. & White, H. (1999), "Data-Snooping, Technical Trading Rule
  Performance, and the Bootstrap", *Journal of Finance* 54(5).** The Reality Check applied to a
  universe of chart-based rules — the closest precedent for section 6.
- **Holm, S. (1979), "A Simple Sequentially Rejective Multiple Test Procedure", *Scandinavian
  Journal of Statistics* 6(2), 65-70.** Family-wise correction across the 126 predictive tests.
- **Campbell, J. Y. & Thompson, S. B. (2008), "Predicting Excess Stock Returns Out of Sample: Can
  Anything Beat the Historical Average?", *Review of Financial Studies* 21(4).** The
  out-of-sample R² against the historical mean used here.
- **Welch, I. & Goyal, A. (2008), "A Comprehensive Look at the Empirical Performance of Equity
  Premium Prediction", *Review of Financial Studies* 21(4).** Most return predictors fail out of
  sample against the historical mean; the analogue forecaster joins them.

## The lead in section 3 — returns after big run-ups

- **De Bondt, W. F. M. & Thaler, R. (1985), "Does the Stock Market Overreact?", *Journal of
  Finance* 40(3).** Long-horizon reversal in the cross-section.
- **Fama, E. F. & French, K. R. (1988), "Permanent and Temporary Components of Stock Prices",
  *Journal of Political Economy* 96(2).** and **Poterba, J. M. & Summers, L. H. (1988), "Mean
  Reversion in Stock Prices: Evidence and Implications", *Journal of Financial Economics* 22(1).**
  Index-level mean reversion at multi-year horizons — evidence that is famously sensitive to
  whether the 1930s are in the sample, which is also where most "looks like 1929" months live.

## Data provenance

- **Fama-French US market factor (`Mkt-RF`) and one-month T-bill (`RF`)**, monthly, 1926-07 →
  2018-11, from Kenneth French's data library as redistributed inside the `arch` package
  (`arch/data/frenchdata/frenchdata.csv.gz`), read through `quantlab.bundled` and verified
  against a SHA-256 pin. `Mkt-RF + RF` is a **total** return (dividends included), nominal.
- **S&P 500 index**, daily closes, 1990-01-02 → 2022-12-28, shipped inside the `skfolio` 1.8.1
  wheel (`skfolio/datasets/data/sp500_index.csv.gz`), fetched once by
  `quantlab.bundled.fetch_skfolio`, cached under `studies/_cache/bundled/` and SHA-256 pinned. A
  **price** index (no dividends), nominal. December 2022 is incomplete on the tape and dropped.
- No live data source is used; Yahoo! Finance is not touched.

## Neighbours on this desk

- **1000-fourier-cycles** — the same "does a random walk show the same thing?" question asked of
  spectral peaks; this study asks it of whole-path matches and then forecasts from them.
- **06-clockwork-vol** — cycles in the VIX against AR(1) red noise.
- **835-spurious-regression** — random walk regressed on random walk; section 1 here is the
  correlation-of-paths version, followed by the search that makes it worse.
- **343-data-mining-roulette** — the multiple-testing problem in its purest form; section 5 is a
  worked instance on a specific folk forecaster.
