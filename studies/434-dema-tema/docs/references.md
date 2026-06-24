# References & literature map — Study 434 (DEMA & TEMA)

## The claim under test

- **The folk recipe / the inventor.** Patrick Mulloy, *"Smoothing Data with Faster Moving
  Averages"* (**Technical Analysis of Stocks & Commodities**, Feb 1994) introduced **DEMA**,
  and a follow-up (*"Smoothing Data with Less Lag,"* TASC, 1994) extended the idea to **TEMA**.
  The construction: `DEMA = 2·EMA − EMA(EMA)` and `TEMA = 3·EMA − 3·EMA(EMA) + EMA(EMA(EMA))`,
  which algebraically cancel the leading-order phase lag of a single EMA. The trading claim,
  repeated in every charting package's docs (TradingView, MetaTrader, StockCharts): *"a faster,
  more responsive moving average means earlier, better trend signals — and more profit."*
- **The steelman we test.** (H1) the price-vs-line long/flat rule on DEMA/TEMA earns a higher
  **net, excess-of-cash Sharpe** than the same rule on a plain SMA; (H2) the trend rule beats
  buy-and-hold. We test both on 21 years of daily SPY, with one execution lag and one-way costs.

## The real phenomenon behind it — filter lag is a bias-variance trade-off

- **Lag and smoothing are coupled.** A moving average is a low-pass filter; reducing its phase
  lag necessarily widens its pass-band, re-admitting high-frequency variance. See Ehlers,
  *Rocket Science for Traders* (2001) and *Cybernetic Analysis for Stocks and Futures* (2004),
  which derive "zero-lag" filters and are explicit that responsiveness is bought with noise.
  This is the mechanism this study measures: DEMA/TEMA's extra responsiveness shows up as ~2×
  turnover and whipsaw, not extra return.
- **Hull MA (a sibling claim).** Alan Hull's HMA (2005) is another "reduced-lag" line built from
  weighted MAs; the same bias-variance argument applies and is a natural follow-on study.

## Why technical MA rules rarely beat the benchmark

- **Brock, Lakonishok & LeBaron (1992),** *Simple Technical Trading Rules and the Stochastic
  Properties of Stock Returns* (**Journal of Finance**) — moving-average rules looked profitable
  pre-1987 but the result is fragile and largely vanishes out-of-sample / after costs.
- **Sullivan, Timmermann & White (1999),** *Data-Snooping, Technical Trading Rule Performance,
  and the Bootstrap* (**Journal of Finance**) — once you correct for the universe of rules
  searched, technical-rule outperformance is statistically insignificant. The right null for a
  timing rule is a randomised schedule of the same activity (our block-permutation placebo).
- **Park & Irwin (2007),** *What Do We Know About the Profitability of Technical Analysis?*
  (**Journal of Economic Surveys**) — a survey: early positive results decay sharply in modern,
  liquid, post-cost samples.
- **Zakamulin (2017),** *Market Timing with Moving Averages* (Palgrave) — a book-length study
  finding moving-average timing rules add little to no risk-adjusted value on broad US equities
  net of costs, and that the choice of *which* average matters far less than believers think.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (**Econometrica**) —
  [`strategy.hac_t`](../dema_tema/strategy.py), used on the daily excess-over-buy-and-hold
  return and on the paired X-minus-SMA difference.
- **Block / permutation null.** Politis & Romano (1994) stationary bootstrap and the
  circular-block resample — [`strategy.block_permutation_pvalue`](../dema_tema/strategy.py)
  randomises the timing while preserving the activity level and block autocorrelation.
- **Sharpe ratio (excess-of-cash).** Sharpe (1966, 1994); the race is run excess-vs-excess so a
  part-time-in-cash rule is compared fairly to an always-invested benchmark.
- **Reproducibility stamp.** [`quantlab/repro.py`](../../../quantlab/repro.py) — the as-of freeze
  and the content fingerprint each headline run carries.

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), adjusted close (`auto_adjust=True`), SPY back
  to 2005-01-03. The reproducible core and the synthetic positive control run on the
  deterministic [`data.synthetic_panel`](../dema_tema/data.py) generator, never the network.

## Related desk studies

- **[Study 104 — Bollinger-Reversion](../../104-bollinger-reversion/)** — bands vs a random buy;
  the same lesson that the *line/envelope* is rarely the edge.
- **[Study 21 — Fools-Gold](../../21-fools-gold/)** — the 50/200 golden cross; a moving-average
  timing rule that dies against a buy-and-hold baseline.
- **[Study 72 — Loaded-Dice](../../72-loaded-dice/)** — the intraday SMA-crossover scalp; a fair
  coin even on the trending intraday tape.
