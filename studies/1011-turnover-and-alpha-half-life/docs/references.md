# Sources & literature map — Study 1011 (The Half-Life of an Edge)

## The fundamental law

- **Grinold, R. C. (1989), "The Fundamental Law of Active Management", *Journal of Portfolio
  Management* 15(3), 30-37.** IR = IC × √BR.
- **Grinold, R. C. & Kahn, R. N. (1999), *Active Portfolio Management*, 2nd ed., McGraw-Hill.**
  The full treatment, including the transfer coefficient.
- **Clarke, R., de Silva, H. & Thorley, S. (2002), "Portfolio Constraints and the Fundamental
  Law of Active Management", *Financial Analysts Journal* 58(5), 48-66.** Introduces the
  transfer coefficient — the reason predicted IR exceeds realised IR in section 5.
- **Buckle, D. (2004), "How to Calculate Breadth: An Evolution of the Fundamental Law of Active
  Portfolio Management", *Journal of Asset Management* 4(6), 393-405.** The correlation
  correction to breadth applied in `effective_breadth`.
- **Ding, Z. (2010), "The Fundamental Law of Active Management: Time Series Dynamics and
  Cross-Sectional Properties", SSRN 1520262.** Breadth as a function of signal autocorrelation —
  the time-dimension correction used here.

## Trading costs and optimal turnover

- **Gârleanu, N. & Pedersen, L. H. (2013), "Dynamic Trading with Predictable Returns and
  Transaction Costs", *Journal of Finance* 68(6), 2309-2340.** The partial-trading result in
  `gp_trade_rate`: trade a constant fraction toward an aim portfolio, with the fraction set by
  decay and cost.
- **Almgren, R. & Chriss, N. (2001), "Optimal Execution of Portfolio Transactions", *Journal of
  Risk* 3(2), 5-40.**
- **Novy-Marx, R. & Velikov, M. (2016), "A Taxonomy of Anomalies and Their Trading Costs",
  *Review of Financial Studies* 29(1), 104-147.** Which anomalies survive costs, organised by
  turnover — the empirical counterpart to section 8.
- **Frazzini, A., Israel, R. & Moskowitz, T. J. (2018), "Trading Costs", SSRN 3229719.** Real
  execution costs from a large manager's own data, much lower than academic estimates.

## Signal decay

- **Chan, L. K. C., Jegadeesh, N. & Lakonishok, J. (1996), "Momentum Strategies", *Journal of
  Finance* 51(5), 1681-1713.**
- **Jegadeesh, N. (1990), "Evidence of Predictable Behavior of Security Returns", *Journal of
  Finance* 45(3), 881-898.** Short-term reversal, the fast-decaying signal here.
- **McLean, R. D. & Pontiff, J. (2016), "Does Academic Research Destroy Stock Return
  Predictability?", *Journal of Finance* 71(1), 5-32.** Decay of a different kind — after
  publication — and a reminder that a half-life measured in-sample is an upper bound.

## Neighbours on this desk

**1001-purged-cv-embargo**, **997-rebalance-timing-luck**, **1010-correlation-matrix-stability**,
**860-backtest-overfitting**.
