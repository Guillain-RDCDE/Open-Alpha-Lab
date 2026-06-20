# References & literature map — Study 344 (Backtest-Overfitting)

## The claim under test

- **The disease.** Bailey, Borwein, López de Prado & Zhu (2014), *Pseudo-Mathematics and
  Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance*
  (Notices of the American Mathematical Society, 61(5)). The foundational statement: if you try
  enough strategy configurations on the same data, you *will* find a great-looking backtest, and the
  expected maximum in-sample Sharpe grows without bound in the number of trials — so a high Sharpe
  with an undisclosed trial count is meaningless. This is the steelman we reproduce: *every backtest
  looks gorgeous because the search itself manufactures the beauty.*
- **Why live runs disappoint.** Harvey & Liu (2014), *Backtesting* (Journal of Portfolio
  Management) and Harvey, Liu & Zhu (2016), *... and the Cross-Section of Expected Returns* (Review
  of Financial Studies) — the multiple-testing critique of the published factor zoo: with hundreds of
  tested factors, the conventional *t* > 2 bar is far too lax; a haircut is required.

## The two diagnostics implemented here

- **Deflated Sharpe Ratio (DSR).** Bailey & López de Prado (2014), *The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting and Non-Normality* (Journal of Portfolio
  Management, 40(5)). The DSR re-expresses a Sharpe as the probability it exceeds the *expected
  maximum Sharpe under the null* over N trials, correcting for sample length and the skew/kurtosis of
  returns — implemented in [`strategy.deflated_sharpe_ratio`](../backtest_overfitting/strategy.py)
  with the expected-max formula in [`strategy.expected_max_sharpe`](../backtest_overfitting/strategy.py).
- **Probability of Backtest Overfitting (PBO).** Bailey, Borwein, López de Prado & Zhu (2017), *The
  Probability of Backtest Overfitting* (Journal of Computational Finance, 20(4)). Combinatorially-
  Symmetric Cross-Validation (CSCV): over all balanced in-sample/out-of-sample block partitions, how
  often does the in-sample champion land in the bottom half out-of-sample? Implemented in
  [`strategy.pbo_cscv`](../backtest_overfitting/strategy.py).

## The Sharpe-ratio standard errors

- **Sharpe-ratio sampling distribution.** Lo (2002), *The Statistics of Sharpe Ratios* (Financial
  Analysts Journal) — the higher-moment (skew/kurtosis) correction to the Sharpe-ratio standard
  error that the DSR denominator uses. Mertens (2002) gives the equivalent variance expression.
- **Multiple-comparisons context.** The expected-maximum-of-N-Gaussians asymptotics behind
  `expected_max_sharpe` follow the extreme-value approximation in Bailey & López de Prado (2014),
  using the Euler-Mascheroni constant.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) — used here to
  ask whether the *selected* champion's out-of-sample mean differs from zero
  ([`strategy.hac_tstat`](../backtest_overfitting/strategy.py)).
- **Circular block bootstrap.** Politis & Romano (1992), *A circular block-resampling procedure for
  stationary data* — preserves autocorrelation when CI-ing the mean return
  ([`strategy.block_bootstrap_ci`](../backtest_overfitting/strategy.py)).
- **White's Reality Check & the SPA test.** White (2000), *A Reality Check for Data Snooping*
  (Econometrica); Hansen (2005), *A Test for Superior Predictive Ability* — the alternative
  data-snooping corrections the desk's protocol names. DSR and PBO are the trial-count- and
  cross-validation-based cousins of the same idea.

## Data sources used here

- **Yahoo! Finance** (via `yfinance` / the shared `quantlab.data` loader), SPY total-return daily
  closes, 2000–2026, for the single real worked example. All headline numbers are pinned with an
  as-of date (2026-05-31, the last full month) and content fingerprints (see
  [`docs/results.md`](results.md)). The offline reproducible core and the entire test-suite run on
  the deterministic [`data.synthetic_prices`](../backtest_overfitting/data.py) generator, never the
  network.

## Related desk studies

- **[Study 350 — Dartboard-Portfolio](../../350-dartboard-portfolio/)** and the rest of the
  *research-method demos*: companion teardowns of how a benign-looking procedure manufactures a
  result. Study 344 is the meta-study — it explains *why the whole desk insists on robust inference,
  trial-count disclosure, and out-of-sample discipline* in the first place.
- **[Study 97 — Balancing-Act](../../97-balancing-act/)**: same desk machinery (HAC *t*,
  block-bootstrap CIs) applied to an allocation rule.
