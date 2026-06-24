# References & literature map — Study 439 (Linear Regression Channel)

## The claim under test

- **The folk recipe.** The **Linear Regression Channel** (LRC) ships in every charting platform —
  TradingView, MetaTrader, ThinkorSwim, StockCharts. The standard write-up (TradingView's LRC
  docs; Investopedia, *"Linear Regression Channel"*; countless YouTube tutorials) claims the
  least-squares **slope** of recent prices is a *smoother, leading* read on the trend than a
  lagging moving average: go long when the slope turns up, flat/short when it turns down, and you
  ride trends better than a moving-average crossover. We steelman this to its sharpest testable
  form — *the regression-slope timing rule beats both buy-and-hold and the equivalent moving-
  average rule, net of costs, on a liquid index* — and measure it over 21 years of SPY.

## The real effect the claim leans on — trend / time-series momentum

- **Time-series momentum.** Moskowitz, Ooi & Pedersen (2012), *Time Series Momentum* (Journal of
  Financial Economics) — a security's own past 12-month return predicts its next-month return
  across 58 instruments. A regression slope is one estimator of exactly this trend, so where TSMOM
  is strong (futures, FX, commodities) a slope filter can earn its keep.
- **Moving-average trend rules.** Brock, Lakonishok & LeBaron (1992), *Simple Technical Trading
  Rules and the Stochastic Properties of Stock Returns* (Journal of Finance) — the canonical study
  of MA crossover rules; high in-sample t-stats that the data-snooping literature later deflated.
  The LRC slope is a close cousin of these MA rules (see the method note below).
- **Why a slope is a moving average in disguise.** A rolling OLS slope on an evenly-spaced index is
  a fixed **linear filter** of past prices: slope = Σ wₖ·pₜ₋ₖ with weights wₖ = (k − k̄) /
  Σ(k − k̄)². It is therefore a weighted *difference* of recent prices — structurally a relative of
  the moving average and of MACD (the difference of two EMAs). "Fit a regression" sounds more
  rigorous than "take an average," but on this tape it buys nothing the average didn't.

## Why the steelman fails here — and the trap it hides

- **Bull-market drift inflates every long-biased rule.** SPY total-return compounded ~11%/yr over
  2005–2026. Almost any rule that is long most of the time *makes money*; that is not evidence it
  beats the market. The correct bar is the **active** return over buy-and-hold — and the slope
  rule's is HAC t = −1.51 (negative, insignificant).
- **Sharpe-trimming ≠ alpha.** Sitting in cash ~28% of the time lowers volatility, nudging Sharpe
  up (0.71 vs 0.65) while *lowering* total return (7.8% vs 11.1%/yr). A higher Sharpe from
  de-risking is not the same as out-performing — the head-to-head t is what settles it.
- **Out-of-sample / cross-instrument decay.** Sullivan, Timmermann & White (1999), *Data-Snooping,
  Technical Trading Rule Performance, and the Bootstrap* (Journal of Finance), and Park & Irwin
  (2007), *What Do We Know About the Profitability of Technical Analysis?* (Journal of Economic
  Surveys) — technical-rule edges on a single long bull market routinely vanish out-of-sample or
  across markets. The slope rule beats buy-and-hold on only 1 of 6 names here.

## Method lineage (the desk's shared engine)

- **OLS slope, closed form.** The evenly-spaced regression slope reduces to a constant-denominator
  linear filter, Σ(k−k̄)(p−p̄) / [W(W²−1)/12] — implemented exactly in
  [`strategy.rolling_slope`](../linear_regression_channel/strategy.py) (no per-window lstsq).
- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) — the
  inference-bar statistic on the daily active return, in
  [`strategy.hac_t`](../linear_regression_channel/strategy.py).
- **Block permutation placebo.** Politis & Romano (1994) stationary/block bootstrap logic — we
  block-shuffle the position series (21-day blocks) to preserve its persistence while destroying
  its alignment with returns, in
  [`strategy.permutation_pvalue`](../linear_regression_channel/strategy.py).
- **One execution lag.** Signal at close t earns return t+1 — a single `shift`, applied once
  (`backtest`), per the desk's documented-lag rule.
- **Reproducibility stamp.** As-of freeze + content fingerprint each headline run carries
  ([`data.fingerprint`](../linear_regression_channel/data.py)).

## Data sources used here

- **Yahoo! Finance daily bars** via `yfinance`, auto-adjusted (total-return) closes, 2005-01-03
  onward for SPY (headline) and QQQ/AAPL/MSFT/JPM/XLE (panel). The offline reproducible core, the
  notebooks' synthetic control, and `verify.py` (cache-first) run with no network once the
  parquets under `_cache/` exist; the deterministic
  [`data.synthetic_panel`](../linear_regression_channel/data.py) generator is the positive-control
  tape and never touches the network.

## Related desk studies

- **[Study 104 — Bollinger-Reversion](../../104-bollinger-reversion/)**: the LRC's ±σ *bands* as a
  mean-reversion entry — the channel's *other* claim (this study isolates the *slope*/trend claim).
- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: the 50/200 golden cross — the same "the trend
  is your friend" family; also looks good in a bull market and dies against a buy-and-hold baseline.
- **[Study 78 — Crossed-Wires](../../78-crossed-wires/)**: an MA/indicator rule study that
  separates the mechanical from the informative component of a technical signal.
- **[Study 72 — Loaded-Dice](../../72-loaded-dice/)**: the intraday SMA-crossover cousin — a fair
  coin even on the trending intraday tape.
