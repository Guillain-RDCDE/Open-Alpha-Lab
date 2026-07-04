# References — Study 625 (Starting-Yield-Bond-Decade)

## The claim's source

- **Bogle, J. C. (1991).** "Investing in the 1990s: Occam's Razor Revisited." *Journal of
  Portfolio Management*, 18(1). The origin of the popular form of the claim: the **initial
  yield** explains the overwhelming share of subsequent 10-year bond returns — Bogle reported
  correlations near 0.95 (R² ≈ 0.9) for holding-period bond returns vs the entry yield.
- **Bogle, J. C. & Nolan, M. W. (2015).** "Occam's Razor Redux: Establishing Reasonable
  Expectations for Financial Market Returns." *Journal of Portfolio Management*, 42(1), 119–134.
  The updated version: "the entry yield on the 10-year Treasury explains 92% of the variation
  in its subsequent decade return."
- **Leibowitz, M. L. & Homer, S. (1972/2013).** *Inside the Yield Book*. The rolling-yield
  arithmetic underneath the claim.
- **Leibowitz, M. L., Bova, A. & Kogelman, S. (2014).** "Long-Term Bond Returns under
  Duration Targeting." *Financial Analysts Journal*, 70(1), 31–51. The formal convergence
  result: a duration-targeted (constant-maturity) bond portfolio's annualised return converges
  to its **starting yield** over roughly 2 × duration − 1 years — the "duration arithmetic,
  not forecasting" mechanism this study tests.

## Method

- **Swinkels, L. (2019).** "Treasury Bond Return Data Starting in 1962." *Data*, 4(3), 91.
  The constant-maturity simulation method we use: monthly roll of a par bond priced in closed
  form at the new yield with one month less maturity — validated by Swinkels against actual
  Treasury index returns (correlations ≈ 0.99).
- **Newey, W. K. & West, K. D. (1987).** "A Simple, Positive Semi-Definite, Heteroskedasticity
  and Autocorrelation Consistent Covariance Matrix." *Econometrica*, 55(3), 703–708. The HAC
  correction for the overlapping-window secondary regression (120 lags).
- **Valkanov, R. (2003).** "Long-Horizon Regressions: Theoretical Results and Applications."
  *Journal of Financial Economics*, 68(2), 201–232. Why overlapping long-horizon t-stats are
  unreliable — the reason our primary unit is the **non-overlapping decade**.

## Siblings on this bench

- **[Study 120 — Excess-CAPE-Yield](../../120-excess-cape-yield/)** — the *same decade-unit
  design* on the stock side (Real, unit = decade): plain 1/CAPE fails to predict nominal stock
  decades until the bond yield is netted out (ECY, R² 0.70 on excess returns). This study is
  its bond-side sibling: for bonds the starting yield alone *is* the decade, no repair needed —
  and our third axis shows the trick does **not** transfer to stocks as-is.
- **[Study 151 — Stocks-for-the-Long-Run](../../151-stocks-for-long-run/)** — the long-horizon
  Shiller tape used across the bench.

## Data

- **Shiller, R. J.** — *Irrational Exuberance* long monthly dataset (S&P composite, dividends,
  CPI, GS10 "Long Interest Rate" 1871→present):
  http://www.econ.yale.edu/~shiller/data.htm — read via the staged GitHub mirror
  https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv (columns: SP500,
  Dividend, Earnings, CPI, Long Interest Rate, Real Price, Real Dividend, PE10).
- **CBOE 10-Year Treasury Note Yield Index (^TNX)** via Yahoo Finance / yfinance — monthly mean
  of daily closes, used to extend GS10 from 2023-10 to 2026-06 (same monthly-average convention
  as Shiller's series).

*The synthetic world is a machinery proof (a seeded AR(1) yield path pushed through the same
pricing engine, with a tunable mechanics/noise blend); it is never cited as market evidence.*
