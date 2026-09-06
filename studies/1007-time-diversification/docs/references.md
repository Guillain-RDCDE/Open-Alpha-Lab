# Sources & literature map — Study 1007 (Time Does Not Diversify)

## The original argument

- **Samuelson, P. A. (1963), "Risk and Uncertainty: A Fallacy of Large Numbers", *Scientia* 98,
  108-113.** The founding objection: repeating a favourable gamble does not make it safer.
- **Samuelson, P. A. (1969), "Lifetime Portfolio Selection by Dynamic Stochastic Programming",
  *Review of Economics and Statistics* 51(3), 239-246.** The theorem checked numerically in
  section 7: under CRRA and i.i.d. returns, the optimal share is horizon-independent.
- **Samuelson, P. A. (1994), "The Long-Term Case for Equities", *Journal of Portfolio
  Management* 21(1), 15-24.** His own later, more nuanced statement.
- **Merton, R. C. (1969), "Lifetime Portfolio Selection under Uncertainty: The
  Continuous-Time Case", *Review of Economics and Statistics* 51(3), 247-257.**
- **Bodie, Z. (1995), "On the Risk of Stocks in the Long Run", *Financial Analysts Journal*
  51(3), 18-22.** The option-pricing argument: if stocks got safer with time, the cost of
  insuring them would fall with maturity. It rises.

## The other side

- **Siegel, J. J. (2014), *Stocks for the Long Run*, 5th ed., McGraw-Hill.** The best-known
  statement of the convergence case, with the long-horizon annualised-dispersion chart.
- **Campbell, J. Y. & Viceira, L. M. (2002), *Strategic Asset Allocation*, Oxford UP.** Shows
  that *predictable* returns do make horizon matter — the serious version of the argument, and
  the one section 3 tests for.
- **Barberis, N. (2000), "Investing for the Long Run when Returns Are Predictable", *Journal of
  Finance* 55(1), 225-264.** Horizon effects survive but shrink sharply once parameter
  uncertainty is admitted.

## Mean reversion, and testing for it

- **Lo, A. W. & MacKinlay, A. C. (1988), "Stock Market Prices Do Not Follow Random Walks",
  *Review of Financial Studies* 1(1), 41-66.** The variance-ratio test and its
  heteroscedasticity-robust standard error, implemented in `variance_ratio`.
- **Poterba, J. M. & Summers, L. H. (1988), "Mean Reversion in Stock Prices", *Journal of
  Financial Economics* 22(1), 27-59.**
- **Fama, E. F. & French, K. R. (1988), "Permanent and Temporary Components of Stock Prices",
  *Journal of Political Economy* 96(2), 246-273.**
- **Richardson, M. (1993), "Temporary Components of Stock Prices: A Skeptical View", *Journal of
  Business & Economic Statistics* 11(2), 199-207.** The power problem with long-horizon tests —
  the reason section 3 relies on a bootstrap.
- **Kritzman, M. (1994), "What Practitioners Need to Know About Time Diversification",
  *Financial Analysts Journal* 50(1), 14-18.**

## Neighbours on this desk

**1008-start-date-lottery**, **1002-best-days-missed**,
**1006-most-stocks-underperform-cash**, **1003-bitcoin-in-a-portfolio**.
