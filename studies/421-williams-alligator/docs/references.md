# References & literature map — Study 421 (Williams Alligator)

## The claim under test

- **Williams (1995/1998).** Bill Williams, *Trading Chaos: Applying Expert Techniques to
  Maximize Your Profits* (Wiley, 1995) and *New Trading Dimensions* (Wiley, 1998). The
  Alligator is the centrepiece of his "chaos theory" trading system: three smoothed moving
  averages of the **median price** (high+low)/2 — the **Jaw** (13-period SMMA shifted 8
  bars), **Teeth** (8-period shifted 5), **Lips** (5-period shifted 3). The metaphor: the
  alligator "sleeps" when the lines are intertwined (no trend — stand aside) and "wakes and
  eats" when they fan out in order (Lips > Teeth > Jaw = uptrend; reverse = downtrend),
  signalling a trend to ride until the lines converge again.
- **The steelman.** The three different look-backs act as a multi-timeframe confirmation
  filter: you commit only when fast, medium and slow trend estimates *agree* on direction
  and *separate* in magnitude. The forward displacement is meant to align each average with
  the price it is "predicting", reducing false flips in choppy markets. Taught in countless
  tutorials and built into TradingView, MetaTrader and every major charting platform.

## What the literature says about indicators of this kind

- **Moving-average timing rules, broadly.** Brock, Lakonishok & LeBaron (1992), *Simple
  Technical Trading Rules and the Stochastic Properties of Stock Returns*, Journal of
  Finance — early evidence of MA-rule profitability in pre-1987 data, much of which did not
  survive later out-of-sample and data-snooping scrutiny.
- **Data-snooping correction.** Sullivan, Timmermann & White (1999), *Data-Snooping,
  Technical Trading Rule Performance, and the Bootstrap*, Journal of Finance — once you
  account for the universe of rules that were tried, the apparent edge of the best technical
  rules largely vanishes. This is the spirit of our permutation placebo and benchmark race.
- **Trend-following / time-series momentum.** Moskowitz, Ooi & Pedersen (2012), *Time
  Series Momentum*, JFE — there *is* a real trend premium across many liquid futures, but it
  is a cross-asset, diversified effect, not something a single displaced-MA fan on one equity
  index reliably captures.
- **Risk reduction vs alpha in MA timing.** Zakamulin (2014), *The Real-Life Performance of
  Market Timing with Moving Average and Time-Series Momentum Rules*, Journal of Asset
  Management — after crediting the cash leg and adjusting for time-in-market, MA timing's
  Sharpe advantage is mostly volatility/drawdown reduction and is often not statistically
  significant on post-2000 data. Exactly our finding: the Alligator de-risks, it does not
  time, and it is beaten by a single moving average.

## Why the test is built the way it is

- **The decisive control is the simplest sibling.** Any rule that exits during sustained
  downtrends will beat buy-and-hold on a volatility-adjusted basis — that is cheap. The
  honest question is whether the Alligator's three-line ceremony beats the *dumbest* trend
  filter, a single SMA(200). It does not (it loses outright), which collapses the
  trend-catching claim. This is the head-to-head the brief demands.
- **The permutation placebo isolates timing from exposure.** A circular-block shuffle of the
  signal keeps the in/out *frequency* but destroys *which* days — if a scrambled version
  beats buy-and-hold about as often as the real one (*p* = 0.15 here), the "timing" is luck.
- **The synthetic positive control is a machinery proof only.** A planted-trend tape shows
  the long/short engine *can* detect trends when they exist (diff-*t* climbs past 4) and
  raises no false alarm on an i.i.d. null — so the real-tape null is genuine. Per the desk's
  inference bar, a synthetic Sharpe never backs a real-tape Signal stamp.

## Related desk studies

- **[Study 110 — Faber-Timing](../../110-faber-timing/)**: the SMA(200) long/flat timing
  rule that *beats* the Alligator here. The canonical single-line trend filter and the
  benchmark this study races against.
- **[Study 106 — Supertrend](../../106-supertrend/)**: another lagging ATR-band flip rule —
  same family of "price crosses a trailing band, flip direction" indicators.
- **[Study 178 — CCI](../../178-cci/)**: Lambert's oscillator turned into an
  overbought/oversold timing rule — same honest-arbiter idiom (HAC *t*, a coin/permutation
  control, costs, one-day lag) applied to a different classic indicator.
- **[Study 104 — Bollinger-Reversion](../../104-bollinger-reversion/)**: a band-based
  technical rule with the same "is it timing or noise?" question.
- **[Study 401 — Signal-Stacking](../../401-signal-stacking/)**: why stacking several weak
  technical filters (as the full *Trading Chaos* system does) rarely manufactures a real
  edge — the natural next test for "Alligator + Awesome Oscillator + fractals".

## Method lineage

- **SMMA / Wilder smoothing.** Welles Wilder, *New Concepts in Technical Trading Systems*
  (1978) — the EWM-with-α=1/n smoothing the Alligator's lines use
  ([`strategy.smma`](../williams_alligator/strategy.py)).
- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica —
  [`strategy.summary`](../williams_alligator/strategy.py) and
  [`strategy.sharpe_diff_tstat`](../williams_alligator/strategy.py).
- **Return-difference t-stat (Sharpe comparison).** Jobson & Korkie (1981), *Performance
  Hypothesis Testing with the Sharpe and Treynor Measures*, Journal of Finance.
- **Block bootstrap / permutation.** Politis & Romano (1994), *The Stationary Bootstrap*,
  JASA — the circular-block shuffle in
  [`strategy.block_permutation_pvalue`](../williams_alligator/strategy.py).

## Data sources

- **SPY daily total-return closes** (via `yfinance`, `auto_adjust=True`), 1993-01-29 to
  2026-06-23. The S&P 500 ETF is the canonical large-cap equity tape for a multi-decade
  trend-rule test; split/dividend adjustment is essential for the buy-and-hold comparison.
- **Cash rate proxy:** a flat 4%/yr (FRED unavailable in this sandbox; a conservative
  long-run proxy for the Fed funds path, which ranged 0%–5.5% over the sample).
