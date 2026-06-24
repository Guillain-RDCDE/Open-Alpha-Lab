# References & literature map — Study 435 (Guppy Multiple MA)

## The claim under test

- **Daryl Guppy** is the Australian trader who created and popularised the **Guppy Multiple
  Moving Average (GMMA)** in the late 1990s. See his books *Trading Tactics* (1997) and
  *Trend Trading* (2004), and his columns/site (guppytraders.com). The indicator plots
  **two ribbons** of exponential moving averages: a short-term "trader" group (3, 5, 8, 10,
  12, 15 days) and a long-term "investor" group (30, 35, 40, 45, 50, 60 days). The folk
  reading: you are in an uptrend when the short ribbon is above the long ribbon; the trend
  has **conviction** (and will continue) when the two ribbons are **wide apart and
  parallel**; the trend is exhausting when they **compress and tangle**.
- **The steelman.** GMMA is not just one crossover — Guppy's claim is that the *relationship
  between the ribbons* carries information a single moving average cannot: the short ribbon
  reflects short-term-trader sentiment, the long ribbon reflects investor conviction, and the
  *spacing* between them measures how strongly the two groups agree. A wide, parallel
  separation is therefore meant to be a higher-quality "stay long" signal than a bare price
  > SMA cross. We test exactly that: does the ribbon's width (the conviction reading) add
  value over the plain cross, and does the whole twelve-line apparatus beat a single line?

## Why a trend filter *can* (sometimes) help — the mechanism literature

- **Time-series momentum / trend-following.** Moskowitz, Ooi & Pedersen (2012), *Time Series
  Momentum*, Journal of Financial Economics — a positive (negative) past 12-month return
  predicts a positive (negative) future return across 58 instruments. A moving-average cross
  is a coarse, binary proxy for this. The GMMA ribbon cross is one such proxy.
- **Moving-average timing rules.** Brock, Lakonishok & LeBaron (1992), *Simple Technical
  Trading Rules and the Stochastic Properties of Stock Returns*, Journal of Finance — the
  classic systematic test of MA-crossover rules; reported pre-1987 outperformance that later
  work attributed largely to data-snooping. Sullivan, Timmermann & White (1999), *Data-Snooping,
  Technical Trading Rule Performance, and the Bootstrap*, Journal of Finance — show that once
  you correct for the universe of rules searched, the apparent MA-rule edge largely vanishes.
- **The Faber binary timing benchmark.** Faber (2007), *A Quantitative Approach to Tactical
  Asset Allocation*, SSRN 962461 — the single-SMA in/out rule the GMMA is, in effect, a
  twelve-line elaboration of. Zakamulin (2014), *The Real-Life Performance of Market Timing
  with Moving Average and Time-Series Momentum Rules*, Journal of Asset Management — after
  crediting the cash leg and matching time-in-market, the MA-timing Sharpe edge is often not
  statistically significant, especially post-2000. Our result is the same story for the GMMA.

## Why GMMA specifically tends to fail the bar

- **More lines ≠ more information.** Twelve EMAs of overlapping spans are nearly collinear:
  the short-ribbon mean and the long-ribbon mean are themselves smoothed moving averages, so
  the "ribbon cross" is operationally a slow-vs-fast MA crossover wearing a costume. The
  desk's own MA-family teardowns (below) repeatedly find that adding parameters to a moving
  average does not add edge.
- **Width is hindsight, not foresight.** A "wide, parallel" ribbon describes a trend that
  has *already* run; conditioning entry on it buys late and (our tape shows) trades worse
  than the plain cross. This is the classic confusion of a *descriptive* statistic (the trend
  was strong) with a *predictive* one (the trend will continue).

## Related desk studies

- **[Study 104 — Bollinger-Reversion](../../104-bollinger-reversion/)** — another retail-staple
  band indicator pinned against a random control; same finding that the "magic" is market
  drift, not the bands.
- **[Study 110 — Faber-Timing](../../110-faber-timing/)** — the single-SMA in/out rule the
  GMMA elaborates; the random-timing control and excess-vs-excess Sharpe race idiom come
  straight from there. Faber's rule earns a real *risk* stamp; the GMMA does not even manage that.
- **[Study 106 — Supertrend](../../106-supertrend/)** and **[Study 103 — Turtle-Trader](../../103-turtle-trader/)**
  — other trend-following timing rules tested under the same gauntlet; useful for calibrating
  what a trend rule that *does* clear the bar looks like.
- **[Study 178 — CCI](../../178-cci/)** — a single-oscillator technical indicator turned into
  a timing rule; the closest cousin in shape to this study.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica —
  [`strategy.summary`](../guppy_mma/strategy.py) (`_hac_tstat`).
- **Return-difference t-stat (Sharpe race).** Jobson & Korkie (1981), *Performance Hypothesis
  Testing with the Sharpe and Treynor Measures*, Journal of Finance —
  [`strategy.sharpe_diff_tstat`](../guppy_mma/strategy.py).
- **Permutation / rotation placebo.** A circular-rotation label-shuffle that preserves the
  signal's autocorrelation and in-market fraction while destroying its alignment with returns
  — [`strategy.permutation_pvalue`](../guppy_mma/strategy.py).

## Data sources

- **SPY daily total-return closes** (via `yfinance`, `auto_adjust=True`), 1993-01-29 →
  2026-06-12. Split- and dividend-adjusted; essential for a multi-decade buy-and-hold
  comparison. The S&P 500 ETF is the canonical liquid index tape.
- **Cash rate proxy:** a flat 4%/yr (FRED/CBOE endpoints unavailable in this sandbox).
  Because every Sharpe is reported excess-of-cash and the race is excess-vs-excess, the cash
  level shifts both arms together and does not change the verdict.
