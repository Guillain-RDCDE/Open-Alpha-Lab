# References & literature map — Study 129 (Heikin-Ashi)

## The claim under test

- **The folk recipe.** A staple of retail charting platforms (TradingView, ThinkorSwim,
  MT4/MT5) and YouTube tutorials: *"Replace your standard candles with Heikin-Ashi —
  the smoothed candles filter the noise and let you see the trend clearly. When the
  colour flips from red to green, go long; from green to red, go short. The smoothing
  keeps you out of whipsaws and on the right side of the trend."* There is no canonical
  academic paper behind the retail claim — it is trading-platform lore. We steelman it
  as the sharpest testable version: *the HA colour-flip direction carries more
  directional information than a random entry, measured on a symmetric-barrier backtest
  that is impervious to exit-ratio manipulation.*

## The construction the claim relies on

- **Heikin-Ashi definition.** The HA transformation is described in Valcu, D. (2004),
  *"Using the Heikin-Ashi technique"*, Technical Analysis of Stocks & Commodities
  Magazine (February 2004). The recursive formula is:
  HA_close = (O+H+L+C)/4; HA_open = (prev_HA_open + prev_HA_close)/2;
  HA_high = max(H, HA_open, HA_close); HA_low = min(L, HA_open, HA_close).
  The recursion means a proper causal implementation requires a bar-by-bar loop —
  a point this study explicitly enforces to prevent look-ahead bias.
- **Smoothing and lag.** The averaging in HA_open introduces a one-bar lag relative to
  the raw price. This is the same trade-off as any moving average: smoothed = fewer
  false signals, but also delayed entry on real signals. Our synthetic positive control
  shows that when momentum *is* present the rule *does* find it, but at the cost of
  entering later into the move — this is the machinery's honest characterisation.

## Why the steelman is almost coherent — what it leans on

- **Return autocorrelation / momentum.** Jegadeesh & Titman (1993), *"Returns to Buying
  Winners and Selling Losers: Implications for Stock Market Efficiency"* (Journal of
  Finance), document cross-sectional momentum at the monthly horizon; Moskowitz, Ooi &
  Pedersen (2012), *"Time Series Momentum"* (Journal of Financial Economics), establish
  time-series momentum in futures. If *any* serial correlation in returns exists at the
  daily horizon, a trend-following indicator like HA could exploit it.
- **Short-horizon return reversal.** Lehmann (1990), *"Fads, Martingales, and Market
  Efficiency"* (Quarterly Journal of Economics), and Jegadeesh (1990), *"Evidence of
  Predictable Behavior of Security Returns"* (Journal of Finance), document short-horizon
  (weekly) reversal effects. If daily returns mean-revert, a trend-following flip signal
  is on the *wrong* side — which is the direction the raw-candle baseline (t = −1.55)
  hints at, though sub-threshold.
- **Technical analysis and the random walk.** Fama (1970), *"Efficient Capital Markets:
  A Review of Theory and Empirical Work"* (Journal of Finance), established the
  benchmark against which all technical rules are judged. Park & Irwin (2007), *"What
  Do We Know About the Profitability of Technical Analysis?"* (Journal of Economic
  Surveys), survey 95 studies: most documented profits disappear after data-snooping
  corrections, transaction costs, and out-of-sample testing.

## The traps this study is about

- **Bull-market drift as pseudo-signal.** The long-side HA trades show t = +3.67, but
  the *random* long control shows t = +3.35 — essentially the same. Over 15 years of
  strong equity bull markets (2011–2026) any strategy that goes long more often than
  short will show a positive mean. The symmetric backtest isolates the *directional*
  claim from the market-direction tailwind.
- **Smoothing as an apparent virtue.** HA produces ~half as many flips as raw candles.
  Retail tutorials present this as "filtering whipsaws" — which sounds like signal
  improvement. But fewer trades means fewer false positives *and* fewer true positives in
  proportion; our comparison with the raw-candle baseline shows the smoothing does not
  improve the directional t-stat versus a coin.
- **The win-rate illusion is also present.** With a symmetric 1:1 barrier, HA's win-rate
  is ~50.7% — essentially a coin. The "HA gives a high win-rate" narrative, common in
  retail tutorials, typically reflects asymmetric exit settings (large target, small stop),
  the same bias exposed in Study 72 (Loaded-Dice).

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *"A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix"* (Econometrica) —
  [`strategy.summarize`](../heikin_ashi/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Sharpe with robust SE / annualisation.** Lo (2002), *"The Statistics of Sharpe
  Ratios"* (Financial Analysts Journal) — [`quantlab.analytics.sharpe_with_se`](../../../quantlab/analytics.py).
- **Block bootstrap CI.** Politis & Romano (1994), *"The Stationary Bootstrap"* (JASA) —
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).
- **Average true range.** Wilder (1978), *New Concepts in Technical Trading Systems* —
  the risk unit R for the symmetric barriers, [`strategy.atr`](../heikin_ashi/strategy.py).

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), 15 years of daily OHLCV across four
  liquid tapes (SPY, QQQ, IWM, AAPL), 2011-06-13 to 2026-06-12. The offline
  reproducible core and the test-suite run on the deterministic
  [`data.synthetic_daily`](../heikin_ashi/data.py) generator, never the network.

## Related desk studies

- **[Study 72 — Loaded-Dice](../../72-loaded-dice/)**: the SMA(5/10) 5-minute crossover
  scalp — same "trend filter vs a coin" family, same null result, same exit-ratio trap.
- **[Study 106 — Supertrend](../../106-supertrend/)**: Supertrend (ATR band indicator)
  on daily bars — a related smoothed-indicator flip, different signal, much stronger
  result (REAL/FRAGILE); a useful contrast to this study's NONE.
- **[Study 78 — Crossed-Wires](../../78-crossed-wires/)**: MACD daily crossover — another
  smoothed indicator in the same family, also tested with symmetric barriers vs a coin.
- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: the daily 50/200 golden cross —
  a moving-average crossover on the daily bar, the same broader "smooth, then flip"
  family.
