# References — Study 118 (Fed-Model)

## Primary sources

1. **Yardeni, E. (1997).** "Fed's Stock Market Model Finds Overvaluation." *Topical Study #38,
   Deutsche Morgan Grenfell.* The original articulation of the model: fair-value P/E equals
   the inverse of the 10-year Treasury yield (E/P = yield in equilibrium).

2. **Asness, C. S. (2003).** "Fight the Fed Model: The Relationship Between Future Returns and
   Stock and Bond Market Yields." *Journal of Portfolio Management, 30(1), 11–24.*
   The decisive academic critique: E/P is a real quantity; comparing it to the nominal yield
   creates a unit mismatch. Asness shows that E/P alone (or E/P vs real rates) is a far better
   predictor than the composite spread.

3. **Campbell, J. Y., & Thompson, S. B. (2008).** "Predicting Excess Stock Returns Out of
   Sample: Can Anything Beat the Historical Average?" *Review of Financial Studies, 21(4),
   1509–1531.* Introduces the OOS R² criterion and shows most predictors — including yield-based
   ones — fail to beat the historical mean out of sample.

4. **Shiller, R. J. (2000).** *Irrational Exuberance.* Princeton University Press.
   The source of the Shiller monthly dataset used here (Earnings, SP500, CPI, Long Rate, Real
   Price, Real Dividend, PE10). Dataset available at http://www.econ.yale.edu/~shiller/data.htm.

## Econometric methods

5. **Newey, W. K., & West, K. D. (1987).** "A Simple, Positive Semi-Definite,
   Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica, 55(3),
   703–708.* The HAC standard error used to correct for serial overlap in forward returns.

6. **Hodrick, R. J. (1992).** "Dividend Yields and Expected Stock Returns: Alternative
   Procedures for Inference and Measurement." *Review of Financial Studies, 5(3), 357–386.*
   Shows that the standard t-statistic on overlapping long-horizon regressions is severely
   biased upward; recommends using non-overlapping data or correcting with long-lag HAC.

## Related work on return predictability

7. **Goyal, A., & Welch, I. (2008).** "A Comprehensive Look at The Empirical Performance of
   Equity Premium Prediction." *Review of Financial Studies, 21(4), 1455–1508.*
   Comprehensive OOS study of equity-premium predictors (including E/P and the dividend yield).
   Most predictors fail OOS; E/P is marginally better but not robustly so.

8. **Fama, E. F., & French, K. R. (1988).** "Dividend Yields and Expected Stock Returns."
   *Journal of Financial Economics, 22(1), 3–25.* The foundational paper on long-horizon
   return predictability; shows high (dividend / price) forecasts high future returns, but
   nearly all the predictability is at 3–5 year horizons.

9. **Cochrane, J. H. (2011).** "Presidential Address: Discount Rates." *Journal of Finance,
   66(4), 1047–1108.* Reviews the evidence that expected returns vary over time, with the
   yield spread as a key state variable; puts the Fed-Model evidence in the broader context of
   discount-rate variation.
