# Sources & literature map — Study 1025 (Don't Look)

## The claim

- **Benartzi, S. & Thaler, R. H. (1995), "Myopic Loss Aversion and the Equity Premium Puzzle",
  *Quarterly Journal of Economics* 110(1), 73-92.** The claim under test: loss aversion plus
  frequent evaluation makes stocks unattractive, and with prospect-theory preferences investors
  are indifferent between stocks and bonds at an evaluation period of about one year.
- **Tversky, A. & Kahneman, D. (1992), "Advances in Prospect Theory: Cumulative Representation
  of Uncertainty", *Journal of Risk and Uncertainty* 5(4), 297-323.** The preferences:
  α = 0.88, λ = 2.25, probability-weighting parameters 0.61 (gains) and 0.69 (losses) — used
  unchanged in this study's headline.
- **Kahneman, D. & Tversky, A. (1979), "Prospect Theory: An Analysis of Decision under Risk",
  *Econometrica* 47(2), 263-291.** The original value function and reference dependence.
- **Mehra, R. & Prescott, E. C. (1985), "The Equity Premium: A Puzzle", *Journal of Monetary
  Economics* 15(2), 145-161.** The puzzle that myopic loss aversion sets out to explain.

## Experimental support

- **Thaler, R. H., Tversky, A., Kahneman, D. & Schwartz, A. (1997), "The Effect of Myopia and
  Loss Aversion on Risk Taking: An Experimental Test", *Quarterly Journal of Economics* 112(2),
  647-661.** Subjects who saw outcomes more often invested less in the risky fund.
- **Gneezy, U. & Potters, J. (1997), "An Experiment on Risk Taking and Evaluation Periods",
  *Quarterly Journal of Economics* 112(2), 631-645.** Less frequent feedback, more risk taken.
- **Benartzi, S. & Thaler, R. H. (1999), "Risk Aversion or Myopia? Choices in Repeated Gambles
  and Retirement Investments", *Management Science* 45(3), 364-381.** Showing the distribution
  of long-horizon returns raises the equity share people choose.
- **Haigh, M. S. & List, J. A. (2005), "Do Professional Traders Exhibit Myopic Loss Aversion?
  An Experimental Analysis", *Journal of Finance* 60(1), 523-534.** Professional traders show
  the effect at least as strongly as students.

## Equilibrium and horizon

- **Barberis, N., Huang, M. & Santos, T. (2001), "Prospect Theory and Asset Prices", *Quarterly
  Journal of Economics* 116(1), 1-53.** Puts loss aversion into a general-equilibrium model.
  Benartzi-Thaler's comparison is partial-equilibrium, and so is this study's.
- **Samuelson, P. A. (1963), "Risk and Uncertainty: A Fallacy of Large Numbers", *Scientia*.**
  The original argument that a horizon does not change the attractiveness of a repeated bet
  under standard preferences. Myopic loss aversion is one of the preference structures under
  which the horizon *does* matter.
- **Poterba, J. M. & Summers, L. H. (1988), "Mean Reversion in Stock Prices: Evidence and
  Implications", *Journal of Financial Economics* 22(1), 27-59.** and **Lo, A. W. & MacKinlay,
  A. C. (1988), "Stock Market Prices Do Not Follow Random Walks: Evidence from a Simple
  Specification Test", *Review of Financial Studies* 1(1), 41-66.** Variance ratios. Here they
  explain why the break-even horizon on the real sequence of returns differs from the i.i.d.
  one (section 2 of the results).

## The switcher

- **Moskowitz, T. J., Ooi, Y. H. & Pedersen, L. H. (2012), "Time Series Momentum", *Journal of
  Financial Economics* 104(2), 228-250.** The myopic switcher ("bills after a loss, stocks after
  a gain") is a one-period sign rule, so time-series momentum is its closest systematic
  relative. Slow clocks lose little because of exactly this kind of persistence.

## Method

- **Künsch, H. R. (1989), "The Jackknife and the Bootstrap for General Stationary
  Observations", *Annals of Statistics* 17(3), 1217-1241.** Block bootstrap.
- **Politis, D. N. & Romano, J. P. (1994), "The Stationary Bootstrap", *Journal of the American
  Statistical Association* 89(428), 1303-1313.**
- **Newey, W. K. & West, K. D. (1987), "A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix", *Econometrica* 55(3), 703-708.** HAC t-stats on
  the switcher's return gap.

## Data provenance

All tapes are read through [`quantlab/bundled.py`](../../../quantlab/bundled.py) and checked
against SHA-256 pins. Nothing is fetched at run time.

- **Fama-French factors** (Kenneth R. French Data Library), as shipped in the `arch` package
  (`arch/data/frenchdata`): `Mkt-RF + RF` is the stock leg (**total return**) and `RF` the bill
  leg. Monthly, 1926-07 → 2018-11.
- **Moody's seasoned AAA corporate bond yield** (as distributed by FRED), in `arch`
  (`arch/data/default`). Monthly average, percent, 1919-01 → 2018-12. The bond leg is
  **constructed** from it (duration-convexity approximation, 20-year constant-maturity par
  bond), which is documented in `dontlook/data.py`.
- **US core CPI** (`CPILFESL`, FRED) in `arch` (`arch/data/core_cpi`), 1957-01 → 2018-11. This is
  core CPI, ex food and energy. Headline CPI is not in the bundle.
- **S&P 500 price index**, daily, from the `skfolio` 1.8.1 wheel (`sp500_index`), 1990-01-02 →
  2022-12-28. **Price only**, no dividends. It is cut at the 2018-11-30 as-of.

## Neighbours on this desk

**1007-time-diversification** (dispersion and horizon), **151-stocks-for-long-run** (does
equity always win over the long run), **1008-start-date-lottery** (start-date path dependence),
**1002-best-days-missed** (the cost of missing particular days), **332-downside-beta** (pricing
of downside risk in the cross-section).
