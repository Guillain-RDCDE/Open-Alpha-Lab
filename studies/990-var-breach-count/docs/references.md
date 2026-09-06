# Sources & literature map — Study 990 (Counting the Breaks)

## The tests

- **Kupiec, P. H. (1995), "Techniques for Verifying the Accuracy of Risk Measurement Models",
  *Journal of Derivatives* 3(2), 73-84.** The unconditional-coverage likelihood-ratio test, and
  — importantly for this study — the original power calculations showing how large a sample it
  needs.
- **Christoffersen, P. F. (1998), "Evaluating Interval Forecasts", *International Economic
  Review* 39(4), 841-862.** The independence test and the joint conditional-coverage test. The
  half of the apparatus most practitioners skip.
- **Christoffersen, P. F. & Pelletier, D. (2004), "Backtesting Value-at-Risk: A Duration-Based
  Approach", *Journal of Financial Econometrics* 2(1), 84-108.** Tests based on the time
  *between* breaches, which catch clustering the lag-one Markov test misses — the main
  extension this study does not implement.
- **Escanciano, J. C. & Olmo, J. (2010), "Backtesting Parametric Value-at-Risk with Estimation
  Risk", *Journal of Business & Economic Statistics* 28(1), 36-51.** Why treating the VaR
  forecast as a known number makes these tests over-reject.

## The models

- **J.P. Morgan/Reuters (1996), *RiskMetrics — Technical Document*, 4th ed.** The EWMA
  variance with λ = 0.94 in `var_ewma`.
- **Barone-Adesi, G., Giannopoulos, K. & Vosper, L. (1999), "VaR without Correlations for
  Portfolios of Derivative Securities", *Journal of Futures Markets* 19(5), 583-602.** Filtered
  historical simulation — the model that combines fat tails with conditioning.
- **Hull, J. & White, A. (1998), "Incorporating Volatility Updating into the Historical
  Simulation Method for Value-at-Risk", *Journal of Risk* 1(1), 5-19.** The same idea arrived at
  independently, and a clearer exposition of why plain historical simulation fails.
- **Pritsker, M. (2006), "The Hidden Dangers of Historical Simulation", *Journal of Banking &
  Finance* 30(2), 561-582.** Exactly how and why the unconditional models cluster their
  breaches.

## Why VaR is the wrong measure anyway

- **Artzner, P., Delbaen, F., Eber, J.-M. & Heath, D. (1999), "Coherent Measures of Risk",
  *Mathematical Finance* 9(3), 203-228.** VaR is not subadditive; expected shortfall is. The
  formal version of section 5's complaint.
- **Basel Committee on Banking Supervision (2016), *Minimum Capital Requirements for Market
  Risk*.** The regulatory move from VaR to expected shortfall, and the traffic-light backtesting
  regime built on breach counts.

## Neighbours on this desk

**214-value-at-risk-basics**, **507-expected-shortfall**, **311-fat-tails**,
**256-volatility-clustering**, **966-garch-vs-har**.
