# Sources & literature map — Study 1001 (The Leaky Fold)

## The fix

- **López de Prado, M. (2018), *Advances in Financial Machine Learning*, Wiley, ch. 7.**
  Purging and embargoing, and the observation that overlapping labels make standard
  cross-validation invalid even without shuffling. `purged_kfold` implements it directly.
- **López de Prado, M. (2018), "The 10 Reasons Most Machine Learning Funds Fail", *Journal of
  Portfolio Management* 44(6), 120-133.** The practitioner summary, with leakage as reason one.

## Why cross-validation fails on dependent data

- **Arlot, S. & Celisse, A. (2010), "A Survey of Cross-Validation Procedures for Model
  Selection", *Statistics Surveys* 4, 40-79.** The general treatment, including the conditions
  under which CV is valid — none of which financial time series satisfy.
- **Bergmeir, C. & Benítez, J. M. (2012), "On the Use of Cross-Validation for Time Series
  Predictor Evaluation", *Information Sciences* 191, 192-213.** Argues blocked CV is acceptable
  for *purely autoregressive* setups, which is a narrower claim than it is usually cited for.
- **Racine, J. (2000), "Consistent Cross-Validatory Model-Selection for Dependent Data: hv-Block
  Cross-Validation", *Journal of Econometrics* 99(1), 39-61.** The hv-block scheme —
  independently derived, essentially the same idea as purging plus embargo, and eighteen years
  earlier.
- **Bergmeir, C., Hyndman, R. J. & Koo, B. (2018), "A Note on the Validity of Cross-Validation
  for Evaluating Autoregressive Time Series Prediction", *Computational Statistics & Data
  Analysis* 120, 70-83.** The careful statement of when it is and is not safe.

## Overfitting in finance more broadly

- **Bailey, D. H., Borwein, J., López de Prado, M. & Zhu, Q. J. (2014), "Pseudo-Mathematics and
  Financial Charlatanism", *Notices of the AMS* 61(5), 458-471.**
- **Harvey, C. R. & Liu, Y. (2015), "Backtesting", *Journal of Portfolio Management* 42(1),
  13-28.**
- **Arnott, R., Harvey, C. R. & Markowitz, H. (2019), "A Backtesting Protocol in the Era of
  Machine Learning", *Journal of Financial Data Science* 1(1), 64-74.** A checklist in which
  leakage is the first item.

## Neighbours on this desk

**996-palindrome-dates**, **860-backtest-overfitting**, **997-rebalance-timing-luck**,
**554-walk-forward-optimisation**.
