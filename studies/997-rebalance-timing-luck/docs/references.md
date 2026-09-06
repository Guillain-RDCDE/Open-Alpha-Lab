# Sources & literature map — Study 997 (The Rebalance Lottery)

## The effect and its fix

- **Blitz, D., van der Grient, B. & van Vliet, P. (2010), "Fundamental Indexation: Rebalancing
  Assumptions and Performance", *Journal of Index Investing* 1(2), 82-88.** The overlapping-
  portfolio construction implemented in `overlapping_portfolios`, and the demonstration that
  rebalance-date choice materially changes measured index performance.
- **Hoffstein, C., Faber, N. & Braun, S. (2020), "Rebalance Timing Luck: The (Dumb) Luck of
  Smart Beta", SSRN 3673910.** The most complete treatment: a formal decomposition of timing
  luck, its scaling with holding period, and why it is largest for high-turnover selection
  rules. The direct source of this study's framing.
- **Hoffstein, C. (2018), "Rebalance Timing Luck: The Difference Between Hired and Fired",
  Newfound Research.** The practitioner version, with the observation that the spread between
  the luckiest and unluckiest variant routinely exceeds a manager's whole track record.

## Why it is a selection problem, not a rebalancing one

- **Jegadeesh, N. & Titman, J. (1993), "Returns to Buying Winners and Selling Losers", *Journal
  of Finance* 48(1), 65-91.** The original momentum paper, which uses *overlapping* portfolios
  precisely for this reason — a detail widely dropped by later replications.
- **Novy-Marx, R. & Velikov, M. (2016), "A Taxonomy of Anomalies and Their Trading Costs",
  *Review of Financial Studies* 29(1), 104-147.** Turnover and implementation choices as
  first-order determinants of a strategy's realised performance.

## Estimation noise in backtests more generally

- **Bailey, D. H., Borwein, J., López de Prado, M. & Zhu, Q. J. (2014), "Pseudo-Mathematics and
  Financial Charlatanism", *Notices of the AMS* 61(5), 458-471.** The general case for reporting
  the distribution rather than the point estimate.
- **Harvey, C. R. & Liu, Y. (2015), "Backtesting", *Journal of Portfolio Management* 42(1),
  13-28.** How much of a reported backtest should be discounted, and for what.

## Neighbours on this desk

**117-rebalancing-bands**, **969-rebalancing-bonus**, **994-small-account-lot-drag**,
**996-palindrome-dates**, **860-backtest-overfitting**.
