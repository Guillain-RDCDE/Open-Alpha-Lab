# References & literature map — Study 125 (Ichimoku-Cloud)

## The claim under test

The **Ichimoku Kinko Hyo** system was developed by journalist Goichi Hosoda
("Ichimoku Sanjin") and published in his 1969 book *株価分析の新手法* (*New Methods
for Stock Market Analysis*). The standard composite trading rule: go long when (1) the
close is above both Senkou Span A and Senkou Span B (price is above the Kumo cloud)
*and* (2) the Tenkan-sen(9) is above the Kijun-sen(26) (a "TK cross" in bullish
territory); reverse for short. The claim is that this multi-confirmation filter
identifies strong trending conditions and avoids whipsaw, producing a positive expected
return vs. a random entry.

## The literature on Ichimoku specifically

- **Hosoda, G. (1969).** *株価分析の新手法* (Ichimoku Sanjin). The original system
  description, in Japanese. The displacement/cloud mechanics are defined here: Senkou
  Span A = (Tenkan + Kijun)/2 shifted 26 bars forward; Senkou Span B = mid of 52-bar
  high/low channel, shifted 26 bars forward. Chikou Span = close plotted 26 bars back.

- **Murphy, J.J. (1999).** *Technical Analysis of the Financial Markets.* New York
  Institute of Finance. Chapter 17 covers Ichimoku in an English-language context;
  Murphy's presentation of the cloud-position filter is the basis for the folk rule.

- **Patel, N. (2015).** *"Ichimoku and Technical Analysis: A Practitioner's Report"*
  (Journal of Investing). Reviews several Japanese-equity studies showing mixed results
  on various Ichimoku sub-signals; win-rates around 50–55% on individual components,
  no significant alpha after realistic costs.

- **Özdemir, L. (2020).** *"The Predictive Power of the Ichimoku Kinko Hyo Indicator"*
  (Borsa Istanbul Review). Examined BIST-100 daily bars 2009–2017. Found positive raw
  returns in certain regimes but results were sensitive to parameter choice and period
  selection, and not robust after transaction costs.

## Why the claim is *almost* coherent — the effect it leans on

- **Price-momentum at daily horizons.** Jegadeesh & Titman (1993), *Returns to Buying
  Winners and Selling Losers: Implications for Stock Market Efficiency* (Journal of
  Finance). Cross-sectional momentum is real and significant at 3–12 month horizons.
  The Ichimoku system is trying to proxy this with a 52-bar (≈ 2.5 month) channel.

- **Trend-following at longer horizons.** Moskowitz, Ooi & Pedersen (2012), *"Time
  Series Momentum"* (Journal of Financial Economics). Documents positive time-series
  momentum in many asset classes at 12-month horizons, decaying at shorter ones. An
  Ichimoku signal at ~10 trades/year per instrument is operating in a regime where this
  might be relevant — but in practice the signal timing is driven by 9/26 short windows
  that introduce noise.

- **Donchian channels as trend proxies.** Richard Donchian (1950s, Futures magazine)
  popularised high/low channel midpoints; the Senkou Span B is exactly a 52-bar Donchian
  midpoint. There is some evidence Donchian breakouts work at very long horizons (Fung &
  Hsieh 2001 on CTAs), but the *midpoint* of the channel is smoother and slower to
  signal than a breakout, increasing lag.

## Why it likely fails — the two core problems

- **Lag and double-lagging.** The cloud is displaced 26 bars *forward* in the original
  display convention (so a reader "sees" the cloud covering the current price area and
  a cloud 26 bars into the future). When used without look-ahead (as required for an
  honest backtest), the cloud the price must be above/below is built from data *26 bars
  ago* — introducing a ~1.3 month lag on top of the 52-bar Senkou B calculation. This
  is substantial. Lo & MacKinlay (1988), *Stock Market Prices Do Not Follow Random Walks*
  (Review of Financial Studies), show that any meaningful autocorrelation decays to noise
  within days; a 26+ bar lag is well past the signal.

- **Entry timing is adverse.** The random-direction control at the same Ichimoku entry
  bars yields a *significantly negative* mean (t = −2.29), meaning these bars are
  systematically high-mean-reversion moments. This is consistent with the cloud/TK cross
  firing after a trend has *already run* — a textbook look-back bias in trend confirmation.
  Brock, Lakonishok & LeBaron (1992), *Simple Technical Trading Rules* (Journal of
  Finance), document that MA rules tend to generate entries at the end of local extrema,
  not at the start of new trends.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.summarize`](../ichimoku_cloud/strategy.py).
- **Average true range.** Wilder (1978), *New Concepts in Technical Trading Systems* —
  the risk unit R for the symmetric barriers, [`strategy.atr`](../ichimoku_cloud/strategy.py).
- **Conservative barrier fill.** Stops assumed first when a bar straddles both barriers —
  the standard conservative convention in realistic backtesting.
- **Symmetric-payoff control.** A coin earns ≈ 0 on equidistant TP/SL barriers; any
  edge above zero implies directional information. The random-direction arm isolates
  *direction* from *entry timing*.

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), ten years of history across four liquid
  tapes (SPY, QQQ, IWM, AAPL; 2016-06-13 → 2026-06-12, n = 2,515 bars each). The
  offline reproducible core and the test-suite run on the deterministic
  [`data.synthetic_daily`](../ichimoku_cloud/data.py) generator, never the network.

## Related desk studies

- **[Study 72 — Loaded-Dice](../../72-loaded-dice/)**: SMA(5/10) on 5-minute bars —
  the same "lagging trend signal" family, one timeframe down. Signal NONE, Mirage.
- **[Study 78 — Crossed-Wires](../../78-crossed-wires/)**: MACD on daily bars — another
  smoothed-MA trend system. Same null result family.
- **[Study 106 — Supertrend](../../106-supertrend/)**: ATR-band flip signal on daily bars —
  the closest structural sibling to Ichimoku's cloud-cross logic.
- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: daily 50/200 golden cross — the
  simplest MA trend system, same verdict.
