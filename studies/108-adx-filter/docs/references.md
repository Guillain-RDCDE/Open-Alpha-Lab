# References & literature map -- Study 108 (ADX-Filter)

## The claim under test

- **The folk rule.** A near-universal retail trading maxim: *"The ADX(14) indicator
  measures trend strength.  Above 25 means the trend is strong enough to trade.  Never
  take a moving-average or breakout signal unless ADX is above 25 -- you'll just get
  whipsawed in choppy markets."*  The rule is popularised by J. Welles Wilder (its
  inventor), codified in virtually every technical-analysis textbook, and repeated
  across charting platforms, trading education sites, and retail strategy forums.  We
  steelman it as the sharpest testable version: *does conditioning an MA(20/50) cross
  signal on ADX(14) >= 25 produce a measurably higher per-trade forward return than the
  unconditioned cross, and does either beat a random-direction control?*

## Why the claim is *almost* coherent -- the real effect ADX tries to capture

- **ADX as a regime filter.** Wilder (1978), *New Concepts in Technical Trading
  Systems* (Trend Research) -- the original source of ADX, +DI, -DI, and the 25-
  threshold rule.  Wilder's intuition: in a ranging (non-trending) market a directional
  entry is on the wrong side of mean reversion roughly half the time; ADX is intended
  to separate trending from ranging regimes so that trend rules are only applied to
  trending markets.
- **Trend-following on daily bars.** Moskowitz, Ooi & Pedersen (2012), *Time Series
  Momentum* (Journal of Financial Economics) -- documents that past 12-month
  performance predicts the next month's direction across many asset classes.  This is
  the real effect that a daily MA cross is a blunt proxy for, and it operates at
  *monthly* horizons -- much slower than a 20/50 cross fires.  Study 21 (Fools-Gold)
  and Study 78 (Crossed-Wires) test the daily cross directly.
- **Regime detection and trend indicators.** Pring (2002), *Technical Analysis
  Explained*, 4th ed. (McGraw-Hill) -- mainstream practitioner treatment of ADX as a
  regime filter; claims the 25-threshold approach reduces false signals in oscillators
  and breakout rules.  The desk's test evaluates this claim directly.

## Why the filter likely adds nothing

- **MA cross is already low-power on daily bars.** Park & Irwin (2007), *What Do We
  Know About the Profitability of Technical Analysis?* (Journal of Economic Surveys) --
  meta-analysis of 95 technical-analysis studies; moving-average profits are largely
  explained by data-snooping bias, transaction costs, and out-of-sample decay.  Our
  UNGATED arm (t = +0.39) is consistent with that: the underlying rule has no
  statistically real edge.
- **Filtering noise out of noise.** If the underlying rule is a coin, applying a
  secondary filter that is itself a lagged function of the same price series cannot
  create a signal that was not there.  ADX is computed from the same OHLCV data as the
  MA; conditioning on it is, in expectation, just a selection-bias on past noise --
  which can easily *reduce* rather than improve forward returns (as observed:
  -41 bps/trade differential).
- **Reduced sample size hurts power without improving expectancy.** With 80% of signals
  filtered away (n = 104 vs 520) the standard error of the per-trade mean increases by
  a factor of sqrt(5), yet there is no improvement in the point estimate.  This is the
  hallmark of a filter that adds no real information.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) --
  [`strategy.summarize`](../adx_filter/strategy.py).
- **Forward-return hold-N engine.** Same architecture as Study 78 (Crossed-Wires) and
  Study 21 (Fools-Gold); enter at next open, close at N-th bar's close.
- **Random-direction control.** The same discipline as Study 72 (Loaded-Dice): the
  only comparison that separates directional signal from the base frequency of winning.
- **ATR computation.** Wilder (1978) -- shared ATR kernel in
  [`strategy.atr`](../adx_filter/strategy.py).
- **Reproducibility stamp.** Content fingerprint and as-of date in
  [`docs/results.md`](results.md).

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`) from 2010-01-04 to 2026-06-12 for six
  liquid names (SPY, QQQ, IWM, AAPL, TSLA, NVDA).  Daily bars go back to 2010 without
  the 60-day cap that applies to sub-hourly data (see Study 72), giving adequate power
  to test a slow (once-per-few-months) signal.  Every headline is pinned with an as-of
  date and a per-tape content fingerprint (see [`docs/results.md`](results.md)); the
  offline reproducible core and tests run on the deterministic
  [`data.synthetic_daily`](../adx_filter/data.py) generator, never the network.

## Related desk studies

- **[Study 21 -- Fools-Gold](../../21-fools-gold/)**: the daily 50/200 golden cross --
  same MA-cross family, slower signal, same null result.
- **[Study 72 -- Loaded-Dice](../../72-loaded-dice/)**: SMA(5/10) on 5-minute bars --
  the intraday variant of the same crossover question, with a random-direction coin as
  the control.
- **[Study 78 -- Crossed-Wires](../../78-crossed-wires/)**: MACD(12,26,9) on daily
  bars -- EMA-based crossover, same daily horizon, same null.
- **[Study 85 -- Dr-Copper](../../85-dr-copper/)**: cross-asset ratio as a market
  regime predictor -- a different family of regime filter ideas; also tested against a
  random baseline.
- **[Study 86 -- Tail-Radar](../../86-tail-radar/)**: VIX-level gating of equity
  trades -- an ADX-analogous "only trade when the regime is right" rule, tested on the
  volatility dimension rather than the trend-strength dimension.
