# Sources & literature map — Study 998 (The Moving Target)

## The filter

- **Kalman, R. E. (1960), "A New Approach to Linear Filtering and Prediction Problems",
  *Journal of Basic Engineering* 82(1), 35-45.** The original.
- **Harvey, A. C. (1989), *Forecasting, Structural Time Series Models and the Kalman Filter*,
  CUP.** The standard econometric treatment, including the random-walk-coefficient model used
  here.
- **Dai, Q. & Rui, X.** — the `delta / (1 - delta)` parameterisation of the state variance is
  the convention popularised in Ernest Chan's implementations, which keeps the tuning parameter
  on an interpretable (0, 1) scale.
- **Chan, E. P. (2013), *Algorithmic Trading: Winning Strategies and Their Rationale*, Wiley,
  ch. 3.** The best-known practitioner exposition of a Kalman-filtered hedge ratio for pairs
  trading, and the direct inspiration for this study's setup.

## Time-varying betas

- **Fama, E. F. & MacBeth, J. D. (1973), *JPE* 81(3), 607-636.** Rolling-window betas, and the
  first systematic treatment of their instability.
- **Adrian, T. & Franzoni, F. (2009), "Learning About Beta: Time-Varying Factor Loadings,
  Expected Returns, and the Conditional CAPM", *Journal of Empirical Finance* 16(4), 537-556.**
  A Kalman filter applied to factor loadings, with the argument that investors themselves are
  filtering.
- **Ghysels, E. (1998), "On Stable Factor Structures in the Pricing of Risk: Do Time-Varying
  Betas Help or Hurt?", *Journal of Finance* 53(2), 549-573.** The dissenting result — a badly
  specified time-varying beta model can be worse than a constant. The reason this study insists
  on a constant-beta control.

## Pairs trading

- **Gatev, E., Goetzmann, W. N. & Rouwenhorst, K. G. (2006), "Pairs Trading: Performance of a
  Relative-Value Arbitrage Rule", *Review of Financial Studies* 19(3), 797-827.** The canonical
  study, and the source of the *z*-score entry/exit convention.
- **Do, B. & Faff, R. (2010), "Does Simple Pairs Trading Still Work?", *Financial Analysts
  Journal* 66(4), 83-95.** The decay of the effect, and why the Sharpe ratios in section 4 are
  small.
- **Avellaneda, M. & Lee, J.-H. (2010), "Statistical Arbitrage in the US Equities Market",
  *Quantitative Finance* 10(7), 761-782.** Mean-reversion speed and the half-life measure used
  in `spread_quality`.

## Neighbours on this desk

**287-pairs-trading**, **604-cointegration-tests**, **973-dimson-beta**,
**987-silver-high-beta-gold**, **992-vol-clustering-halflife**.
