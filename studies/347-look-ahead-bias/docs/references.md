# References & literature map — Study 347 (Look-Ahead-Bias)

## The bias under test

- **Look-ahead bias, defined.** Using information in a backtest that would not have been
  available at the moment of the decision. The canonical form for a close-to-close rule is
  *same-bar execution*: a signal computed from the close of bar `t` is allowed to capture
  the return *of* bar `t`. See the survey treatment in **Bailey, Borwein, López de Prado &
  Zhu (2014), *Pseudo-Mathematics and Financial Charlatanism* (Notices of the AMS)** and
  **López de Prado (2018), *Advances in Financial Machine Learning* (Wiley)**, ch. 11
  ("The Dangers of Backtesting"), which catalogues look-ahead as a primary source of
  backtest overfitting and unreproducible Sharpe ratios.
- **The "seven sins" of quantitative investing.** **Luo et al. (2014, Deutsche Bank
  Quantitative Strategy), *Seven Sins of Quantitative Investing*** names look-ahead /
  survivorship / data-snooping as the recurring ways a paper Sharpe evaporates out of
  sample. Look-ahead is sin #1 because it is the easiest to introduce by accident (an
  un-shifted signal column) and the hardest to spot (the equity curve looks beautiful).
- **Point-in-time data.** Look-ahead also enters through *restated* fundamentals and index
  membership. **Sloan (1996)** and the broad accounting-anomaly literature show effects can
  invert once point-in-time (vs. as-restated) data is used — a fundamental analogue of the
  one-bar price peek studied here.

## Why a peek manufactures Sharpe out of nothing

- **Contemporaneous correlation, not forecasting.** A momentum position `+tanh(z_t)` built
  from the close of `t`, multiplied by the return *of* `t`, is mechanically a function of
  `ret_t` correlated with `ret_t` — a tautology with a large, autocorrelation-robust
  *t*-stat and immunity to transaction costs (the bias is not an exploitable edge that
  spreads can erode). This is the formal reason a look-ahead backtest looks *robust* and
  *cheap to trade* — both are illusions.
- **Data-snooping vs. look-ahead.** Distinct failures: **White (2000), *A Reality Check
  for Data Snooping* (Econometrica)** addresses selection across many strategies;
  look-ahead corrupts a *single* strategy's accounting. A study can be snooping-clean and
  still 100% fantasy if the lag is wrong — which is why the desk fixes the lag *and* runs
  the Reality Check.

## The desk's rule this study motivates

- **One execution lag, documented exactly.** Signal known at the close of `t` earns the
  return of `t+1` — one `shift`, applied once (see `METHODOLOGY.md` → *House rules*). A
  *second* silent shift is just as wrong in the other direction (it once made a one-day
  reversal study measure a two-day-old signal). Calendar-known rules (turn-of-month
  windows) need no lag at all. This study is the empirical case for that rule: it shows the
  exact size of the error a single mis-placed bar produces.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.hac_tstat`](../look_ahead_bias/strategy.py).
- **Circular block bootstrap.** Politis & Romano (1992), *A circular block-resampling
  procedure for stationary data* — preserves autocorrelation when CI-ing the mean return
  ([`strategy.block_bootstrap_ci`](../look_ahead_bias/strategy.py)).

## Data sources used here

- **Yahoo! Finance** (via `yfinance` / the shared `quantlab.data` loader), daily
  total-return adjusted closes for `SPY`, 2005–2026. All headline numbers are pinned with
  an as-of date (2026-05-29, the last full month) and a content fingerprint (see
  [`docs/results.md`](results.md)). The offline reproducible core and the test-suite run on
  the deterministic [`data.synthetic_prices`](../look_ahead_bias/data.py) generator,
  never the network.

## Related desk studies

- **[Study 350 — Dartboard-Portfolio](../../350-dartboard-portfolio/)** and
  **[Study 97 — Balancing-Act](../../97-balancing-act/)** share the engine (excess HAC
  tests, block-bootstrap CIs) and the *research-method-demo* spirit: the verdict is a
  methodology check (here, **CONFIRMED** that a one-bar peek inflates the backtest), not a
  tradable signal.
