# References & literature map — Study 106 (Supertrend)

## The claim under test

- **The folk recipe.** The Supertrend indicator — ATR(10, multiplier 3) — is among
  the most-viewed indicators on TradingView, appearing in millions of published chart
  scripts and embedded in countless retail "algo trading" tutorials.  The recipe: when
  price crosses *above* the upper ATR band, the band locks as a trailing support and the
  signal flips to bullish; when price falls *below* the lower band, the signal flips to
  bearish — enter in the flip direction, ride the resulting trend.  The claim is that
  this *band-lock-and-flip* mechanism extracts a cleaner trend signal than a simple
  moving-average crossover because the band only moves against the trend (ratchets up in
  a bull move, down in a bear), reducing whipsaws.  We steelman it as a testable
  hypothesis: *the Supertrend(10, 3) flip direction carries enough trend information to
  beat a random-direction entry at the same dates, net of costs, on daily bars.*

## Why the steelman is coherent — the real effect it leans on

- **Trend persistence in equity returns (medium-horizon momentum).** Jegadeesh &
  Titman (1993), *Returns to Buying Winners and Selling Losers* (Journal of Finance) —
  the original cross-sectional momentum paper.  The Supertrend is a time-series
  relative of the same effect: it attempts to detect the start of a sustained trend.
  Moskowitz, Ooi & Pedersen (2012), *Time Series Momentum* (Journal of Financial
  Economics), document that single-instrument time-series momentum at the weekly-to-
  monthly horizon has a positive risk premium in equities — precisely the horizon
  the Supertrend's 10-day ATR and rare flips (~8/yr) target.
- **ATR-band trailing stops as trend filters.** Wilder (1978), *New Concepts in
  Technical Trading Systems* — introduced ATR and the original trailing-stop concept
  that Supertrend formalises.  The "volatility normalisation" of using ATR rather than
  a fixed price distance reduces whipsaws in high-volatility regimes, which is a
  sensible engineering choice.
- **The band-lock mechanism.** Lazos, Sutcliffe & Wood (2010, unpublished conference
  version widely circulated) and various TradingView community backtests document the
  lock-and-flip logic that distinguishes Supertrend from a plain HL2±mult×ATR band.
  The ratchet prevents the active band from moving *against* the position, a property
  that reduces false reversals compared to a symmetric ATR channel.

## Comparison to the desk's related studies

- **Study 72 — Loaded-Dice** (SMA(5/10) on 5-minute bars): a null result, HAC t =
  −1.12, vs our real result at t = +3.27.  The difference is **horizon and indicator**:
  5-minute SMA crossovers fire ~11 times per day into noise; Supertrend fires ~8 times
  per *year* into trend structure.  The 5m result says there is no micro-trend at
  the intraday horizon; the daily Supertrend result says there *is* medium-horizon
  trend structure — consistent with the Moskowitz et al. (2012) time-series momentum
  evidence.
- **Study 78 — Crossed-Wires** (MACD(12,26,9) on daily bars): the MACD is another
  lagging EMA-crossover indicator tested at the daily horizon.  A direct comparison
  is instructive: Supertrend(10, 3) fires ~8 times/yr vs MACD's higher turnover, and
  uses volatility-normalised (ATR) bands vs MACD's fixed-width EMA differences.
  Whether the ATR normalisation is the key advantage or the lower turnover (fewer
  whipsaws) is an open question for future desk work.
- **Study 91 — Death-Cross** (50/200 SMA on daily bars): another lagging daily
  crossover; the Death-Cross and Golden-Cross fire even less frequently (~1–2 times/yr)
  than Supertrend — the same "low-frequency trend" family.
- **Study 21 — Fools-Gold** (SMA 50/200 daily): the classical golden-cross family —
  same daily trend-following hypothesis, different indicator lag.

## The two traps this study is mindful of

- **Multiplier specificity / parameter picking.** Only ATR(10, mult=3) clears the
  inference bar; ATR(10, mult=2) and ATR(10, mult=4) are t < 0.5.  The canonical
  TradingView default (10, 3) is the most popular setting in the world — which means
  it is also the most likely to be data-mined.  Park & Irwin (2007), *What Do We Know
  About the Profitability of Technical Analysis?* (Journal of Economic Surveys), document
  how publication bias and parameter search inflate apparent performance.  We report
  this flag explicitly rather than hide it.
- **Bull-market concentration.** The 10-year window (2016–2026) is largely a bull
  market for US equities (SPY, QQQ, AAPL), meaning the long-flip arm (t=+3.63) is
  partly a leveraged bet on a beta-rich period.  The short-flip arm (t=+1.44) is weaker
  and would need a bear-market regime for a fairer evaluation.  Investors looking to run
  this strategy must model drawdown in extended bear markets where short flips will
  underperform.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.summarize`](../supertrend/strategy.py) and [`quantlab.analytics`](../../../quantlab/analytics.py).
- **Sharpe with robust SE / bootstrap CI.** Lo (2002), *The Statistics of Sharpe
  Ratios* (Financial Analysts Journal); Politis & Romano (1994), *The Stationary
  Bootstrap* (JASA) — the CI on the annualised Sharpe.
- **Average true range (RMA/Wilder).** Wilder (1978) — the ATR used both as the
  Supertrend band and as the risk unit R for the symmetric barrier exits.
- **Barrier backtest.** The symmetric ±1 ATR barrier (TP and SL equidistant) is the
  only direction-fair payoff so that a coin scores ≈ 0 and only real directional
  information lifts the mean.  Conservative fill: when a bar straddles both barriers,
  the stop is assumed hit first.

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), 10 years of history (2016-06-13 →
  2026-06-12) across four liquid US equity tickers (SPY, QQQ, IWM, AAPL).  The daily
  tape is long enough for the Supertrend's ~8 flips/year to accumulate ~80 independent
  trades per ticker over the window.  The offline reproducible core and the test-suite
  run on the deterministic [`data.synthetic_daily`](../supertrend/data.py) generator,
  never the network.  Every headline is pinned with an as-of date and a per-tape
  content fingerprint (see [`docs/results.md`](results.md)).
