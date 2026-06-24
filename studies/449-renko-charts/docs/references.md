# References & literature map — Study 449 (Renko-Charts)

## The claim under test

- **The folk recipe.**  Renko charts (from the Japanese *renga*, "brick") plot a new brick
  only when price moves a fixed amount — the *brick size* — in the same or opposite direction,
  discarding time entirely.  The recipe is retold across technical-analysis books and platforms
  (Investopedia, StockCharts ChartSchool, TradingView, ATAS): *"Renko filters out the noise, so
  trends and crossovers are far cleaner and you get fewer false signals than on a time-based
  candlestick chart."*  The sharpest testable form a proponent would accept: a moving-average
  **crossover run on the Renko brick series beats the same crossover run on raw closes** (and,
  at minimum, beats buy-and-hold).  We encode the modern **ATR-Renko** variant (brick = a
  multiple of Average True Range) and a 10/30 SMA crossover.

- **Origin of the chart type.**  Nison, Steve (1994) *Beyond Candlesticks: New Japanese Charting
  Techniques Revealed* (Wiley) — the Western introduction of Renko, three-line-break and kagi
  charts.  Renko is also covered in Hartle, Thom, "Renko Charts" (*Technical Analysis of Stocks
  & Commodities*).

## The real effect the claim leans on — trend / time-series momentum

- **Time-series momentum.**  Moskowitz, Ooi & Pedersen (2012), *Time Series Momentum* (Journal
  of Financial Economics) — trend-following has a real, if modest, premium across asset classes.
  A Renko/MA crossover is one mechanical way to ride it; the question is whether the *brick
  transform* adds anything over a plain crossover.
- **Moving-average rules.**  Brock, Lakonishok & LeBaron (1992), *Simple Technical Trading Rules
  and the Stochastic Properties of Stock Returns* (Journal of Finance) — MA-crossover rules
  showed in-sample profitability that later work attributed largely to data-snooping and
  bull-market drift.
- **Average True Range.**  Wilder, J. Welles (1978), *New Concepts in Technical Trading Systems*
  — the ATR used to scale the brick size, so the brick adapts to each instrument's volatility.

## Why the steelman fails — the attribution is wrong

- **The brick is a redraw, not new information.**  On a daily tape an ATR-sized brick is ~one
  daily range wide, so the step-function Renko close tracks the raw close almost bar-for-bar.
  A 10/30 crossover sees essentially the same series — identical turnover (~7.8 trades/yr) and an
  incremental Sharpe of ~0.  Renko discards *time*, but a daily MA crossover already ignores the
  within-bar path, so there is nothing left to filter.
- **The headline t-stat is bull-market beta.**  Park & Irwin (2007), *What Do We Know About the
  Profitability of Technical Analysis?* (Journal of Economic Surveys), and Sullivan, Timmermann
  & White (1999), *Data-Snooping, Technical Trading Rule Performance, and the Bootstrap* (Journal
  of Finance) — long-only trend rules on a 21-year up-market post high *t*-stats that are mostly
  drift, and routinely **under**-perform buy-and-hold once you account for the time spent in
  cash.  Our crossover (*t* = +3.43) sits below buy-and-hold (*t* = +3.73): no alpha.
- **No look-ahead in a true brick chart.**  A subtle Renko trap is using future bricks to label
  the past; we build the brick level causally (level as of the close of *t*) and apply one
  execution lag, so the comparison is fair.

## The traps this study exposes

- **"Cleaner chart" ≠ "better signal."**  A visually smoother series feels more tradable but
  carries the same information; the eye's comfort is not an edge.  Taleb (2004), *Fooled by
  Randomness*, on confusing the appearance of pattern with predictive content.
- **Choosing the benchmark honestly.**  The right comparison is not "did Renko make money"
  (anything long in a bull market did) but "did Renko beat the *identical* rule without the
  brick, and beat buy-and-hold."  On both counts it does not.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.**  Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.hac_tstat`](../renko_charts/strategy.py).
- **Return-permutation placebo.**  Shuffling daily returns to destroy serial structure is the
  standard non-parametric null for trend rules (cf. White's Reality Check on the stationary
  bootstrap) — [`strategy.permutation_pvalue`](../renko_charts/strategy.py).
- **Reproducibility stamp.**  As-of freeze + content fingerprint each headline run carries
  ([`renko_charts.data.fingerprint`](../renko_charts/data.py)).

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), adjusted close (`auto_adjust=True`,
  total-return), 2005-01-03 to 2026-05-29 for six liquid ETFs: SPY, QQQ, DIA, IWM, EFA, GLD.
  The offline reproducible core and tests run on the deterministic
  [`data.synthetic_panel`](../renko_charts/data.py) generator, never the network.

## Related desk studies

- **[Study 104 — Bollinger-Reversion](../../104-bollinger-reversion/)**: another "the bands/
  indicator hold a secret" technical rule that turns out to be bull-market drift once pinned to a
  fair control — the closest cousin in shape.
- **[Study 178 — CCI](../../178-cci/)**: an oscillator-threshold rule from the same TA-indicator
  zoo, tested against a fair baseline.
- **[Study 343 — Data-Mining-Roulette](../../343-data-mining-roulette/)**: the method demo for why
  a chart that *looks* like it has a pattern usually does not.
