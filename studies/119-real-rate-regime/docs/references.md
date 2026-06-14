# References — Study 119 (Real-Rate-Regime)

## Primary literature

1. **Fama, E. F., & French, K. R. (1989).** "Business conditions and expected returns on
   stocks and bonds." *Journal of Financial Economics*, 25(1), 23–49.
   — Foundational evidence that expected returns vary with economic conditions; dividend
   yield and term spread (related to real rates) predict returns at long horizons.

2. **Campbell, J. Y., & Shiller, R. J. (1988).** "Stock prices, earnings, and expected
   dividends." *Journal of Finance*, 43(3), 661–676.
   — Long-horizon predictability of equity returns; the framework used to construct
   forward real returns from Shiller's real-price series.

3. **Shiller, R. J. (2000).** *Irrational Exuberance.* Princeton University Press.
   — Source of the monthly CAPE/PE10, real price, real dividend, and long-rate series
   used in this study (staged at `_cache/shiller_sp500.parquet`).

4. **Ang, A., & Bekaert, G. (2007).** "Stock return predictability: Is it there?"
   *Review of Financial Studies*, 20(3), 651–707.
   — Systematic evaluation of short-rate and yield-curve based return predictors;
   finds limited predictability at short horizons, more at long horizons.

5. **Ilmanen, A. (2011).** *Expected Returns.* Wiley.
   Chapter 12 ("Fixed income") and Chapter 15 ("Carry in equities") discuss the
   relationship between real rates and equity risk premia. Notes the ambiguity: high
   real rates can reflect either a high discount rate (bearish) or strong real growth
   expectations (bullish).

6. **Damodaran, A. (2023).** "Equity risk premiums: Determinants, estimation and
   implications." *Stern School of Business Working Paper* (annual update).
   — ERP estimates relate real rates to the cost of equity; models with high real rates
   have an ambiguous sign on equity returns depending on the growth assumption.

7. **Newey, W. K., & West, K. D. (1987).** "A simple, positive semi-definite,
   heteroskedasticity and autocorrelation consistent covariance matrix." *Econometrica*,
   55(3), 703–708.
   — The HAC estimator used throughout for inference on overlapping monthly observations.

## On the "Don't fight the Fed" rule

8. **Marty Zweig (1986).** *Winning on Wall Street.* Warner Books.
   — Origin of the phrase "Don't fight the Fed"; advocated stepping aside during Fed
   tightening cycles. Note: Zweig's rule targeted nominal Fed Funds rate cycles, not
   the *real* long rate tested here — an important distinction the popular retelling
   typically ignores.

9. **Edson Gould's "Three Steps and a Stumble"** rule (1970s market commentary).
   — A precursor to the Zweig rule: three consecutive Fed rate hikes signal a bear
   market. Based on nominal Fed Funds moves, not real rates.

10. **Kim, D. H., & Wright, J. H. (2005).** "An arbitrage-free three-factor term structure
    model and the recent behavior of long-term yields and distant-horizon forward rates."
    *Federal Reserve Board Finance and Economics Discussion Series.*
    — Decomposes the long nominal rate into real-rate and inflation components, relevant
    to the construction of the real long rate from Shiller's data.
