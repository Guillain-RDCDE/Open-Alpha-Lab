# References & literature map -- Study 120 (Excess-CAPE-Yield)

## The claim under test

Shiller and Bunn (2014), *The Excess CAPE Yield* (essay, then CFA Institute monograph
chapter), propose that the Cyclically Adjusted Price/Earnings ratio, once the bond-yield
hurdle is subtracted, becomes a direct measure of the equity risk premium and a robust
long-horizon forecaster of the equity-minus-bond excess return over the following decade.
The claim: **ECY = 1/CAPE minus the real 10-year bond yield predicts the forward 10-year
equity excess return with R² near 0.70 on non-overlapping windows, significantly better
than 1/CAPE alone.** This study tests that claim on the full Shiller monthly dataset
(1882--2013 regression sample, 14 non-overlapping 10-year windows).

## Foundational work

- **Campbell & Shiller (1988a).** *The Dividend-Price Ratio and Expectations of Future
  Dividends and Discount Rates.* Review of Financial Studies 1(3): 195--228. The original
  demonstration that the earnings yield predicts long-run returns; ancestor of the whole
  CAPE research programme.
- **Campbell & Shiller (1988b).** *Stock Prices, Earnings, and Expected Dividends.*
  Journal of Finance 43(3): 661--676. Companion paper; introduces the Gordon growth model
  decomposition that links earnings yield to expected returns.
- **Shiller (2000).** *Irrational Exuberance.* Princeton University Press. Popularised CAPE
  (PE10) as a valuation metric and long-run return predictor; the data underlying this study
  are Shiller's own downloadable monthly series.
- **Shiller & Bunn (2014).** *The Excess CAPE Yield.* CFA Institute monograph chapter
  (reprinted in various venues). Defines ECY = 1/CAPE − real bond yield and shows it
  substantially outperforms 1/CAPE alone for forecasting the *excess* return; this study
  replicates and extends their headline result.

## Why the bond-yield adjustment matters

- **Asness (2003).** *Fight the Fed Model.* Journal of Portfolio Management 30(1). Argues
  that while the Fed Model (earnings yield vs nominal bond yield) is commonly used, the
  correct comparison for equity risk premium is the *real* bond yield, not the nominal one.
  ECY uses the real yield (nominal minus trailing CPI).
- **Ilmanen (2011).** *Expected Returns: An Investor's Guide to Harvesting Market Rewards.*
  Wiley. Chapter 4 surveys bond-yield adjustments to valuation measures and their
  predictive power for equity premia; confirms that real-yield-adjusted earnings yield
  dominates the unadjusted version at long horizons.
- **Damodaran (2012, updated annually).** *Equity Risk Premiums: Determinants, Estimation
  and Implications.* Stern NYU working paper. Tracks the implied ERP (a forward-looking
  ECY analogue) and documents its variation with real rates; consistent with ECY's logic.

## Horizon effects and the inference challenge

- **Valkanov (2003).** *Long-Horizon Regressions: Theoretical Results and Applications.*
  Journal of Financial Economics 68(2): 201--232. Shows that OLS t-statistics on
  long-horizon regressions with overlapping observations have *non-standard limiting
  distributions*; the Newey-West correction alone is insufficient. This study reports
  t-statistics on **non-overlapping** 10-year windows to avoid the Valkanov size
  distortion.
- **Hodrick (1992).** *Dividend Yields and Expected Stock Returns: Alternative Procedures
  for Inference and Measurement.* Review of Financial Studies 5(3): 357--386. Introduces
  the reverse-regression alternative to long-horizon overlapping OLS; our non-overlapping
  approach gives qualitatively identical conclusions.
- **Goyal & Welch (2008).** *A Comprehensive Look at the Empirical Performance of Equity
  Premium Prediction.* Review of Financial Studies 21(4): 1455--1508. Finds that most
  predictors, including dividend yield and CAPE variants, underperform a simple historical-
  mean benchmark out of sample. ECY's OOS R² of +0.42 on non-overlapping windows -- though
  with only n=7 test windows -- is a reasonable counter-data-point.

## Tradability: why REAL signal still means MIRAGE

- **Shiller (1981).** *Do Stock Prices Move Too Much to Be Justified by Subsequent Changes
  in Dividends?* American Economic Review 71(3): 421--436. The original paper showing that
  valuation-mean-reversion operates over *decades*, not months or quarters.
- **Siegel (2005).** *Perspectives on the Equity Risk Premium.* Financial Analysts Journal
  61(6): 61--73. Documents that timing on CAPE alone would have had periods of 10--20 years
  underweight equities while they compounded strongly; a 10-year horizon is the unit of
  analysis, not a trading signal.
- **Novy-Marx & Velikov (2016).** *A Taxonomy of Anomalies and Their Trading Costs.* Review
  of Financial Studies 29(1): 104--147. Even anomalies with real signals die at realistic
  transaction costs when turnover is high. ECY's "turnover" is once per decade -- costs are
  negligible -- but the holding period is unacceptable for most institutional mandates.

## Related desk studies

- **[Study 56 -- Tide-Table](../../56-tide-table/)**: CAPE alone as a forecaster of the
  forward *real* equity return (not the excess return); ECY is distinct because it adjusts
  for the bond yield, and this study shows the adjustment is load-bearing.
- **[Study 66 -- Inverted](../../66-inverted/)**: the yield-curve inversion as a macro
  signal; shares the long-rate and inflation data infrastructure.
- **[Study 85 -- Dr-Copper](../../85-dr-copper/)**: copper as a macro lead indicator;
  same Shiller/macro family, different signal.
