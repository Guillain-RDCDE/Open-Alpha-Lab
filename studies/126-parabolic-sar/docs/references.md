# References & literature map — Study 126 (Parabolic-SAR)

## The claim under test

- **Wilder's original recipe.** Welles Wilder Jr. (1978), *New Concepts in Technical
  Trading Systems*, Trend Research.  Chapter 6 introduces the Parabolic SAR: a
  stop-and-reverse system where the trailing stop accelerates as price moves in the
  trend's favour (AF starts at 0.02, steps by 0.02 each bar a new extreme is set, caps
  at 0.20).  Wilder marketed it as a complete system — entry, stop, and exit — and the
  default 0.02/0.20 settings remain the near-universal default on TradingView and every
  major charting platform today.
- **Modern popularisation.** Murphy, J.J. (1999), *Technical Analysis of the Financial
  Markets*, NYIF/Penguin — the standard retail reference.  The Parabolic SAR occupies a
  dedicated chapter as a "time/price" trend-following system.  Constance Brown (2012),
  *Technical Analysis for the Trading Professional*, McGraw-Hill — also covers the SAR
  as a trend confirmation tool.

## Why the steelman is almost coherent — the real effect it leans on

- **Momentum in equities.** Jegadeesh & Titman (1993), *Returns to Buying Winners and
  Selling Losers* (Journal of Finance) — the canonical momentum paper.  At monthly
  horizons momentum in cross-section is robust; the SAR is a time-series trend-follower
  that tries to ride the same persistence.  The connection is legitimate but the horizon
  mismatch is important: daily SAR flips (~20–25/yr) sit at a horizon where the signal
  is much weaker than the 3–12 month cross-sectional effect.
- **Time-series momentum / trend following.** Moskowitz, Ooi & Pedersen (2012),
  *Time Series Momentum* (Journal of Financial Economics) — documents positive 12-month
  autocorrelation in futures prices across asset classes.  Our basket (SPY, QQQ, IWM,
  GLD, TLT, EEM) partially overlaps this universe, which may explain why pooling six
  tickers delivers a marginal positive result.  But the SAR's variable flip frequency
  and the 60-bar horizon of our barrier exit do not directly map to the Moskowitz et al.
  12-month horizon.
- **Trend-following in managed futures / CTAs.** Hurst, Ooi & Pedersen (2017), *A
  Century of Evidence on Trend-Following Investing* (Journal of Portfolio Management) —
  documents that slow trend signals (1–12 months) have historically earned a premium.
  Parabolic SAR on daily bars is a *fast* trend signal; its documented regime is
  explicitly non-choppy markets, which is not guaranteed in our 10-year sample.

## Why the SAR specifically tends to fail in practice

- **Whipsaw in range-bound markets.** The SAR's acceleration factor means it tightens
  rapidly in a trend and generates many false flips in sideways action.  Colby (2003),
  *The Encyclopedia of Technical Market Indicators*, McGraw-Hill — provides a systematic
  review showing the SAR underperforms in range-bound regimes.  Our results (win-rate
  ~53% gross, but only ~2.1%/yr) are consistent with this: a small positive trend bias
  muted by frequent whipsaws.
- **Turnover kills fast technical signals.** Novy-Marx & Velikov (2016), *A Taxonomy of
  Anomalies and Their Trading Costs* (Review of Financial Studies) — any signal
  generating ~22 round-trips/yr faces a very high break-even cost hurdle.  Our desk's
  [Study 72 — Loaded-Dice](../../72-loaded-dice/) (SMA crossover, ~11 trades/day) is
  the intraday extreme of this family; the SAR is far cheaper but still turnover-limited.
- **Data-snooping and parameter sensitivity.** Park & Irwin (2007), *What Do We Know
  About the Profitability of Technical Analysis?* (Journal of Economic Surveys) —
  extensively documents how technical indicator results shrink or vanish when (a)
  parameters are varied, (b) out-of-sample periods are used, and (c) transaction costs
  are applied.  The SAR's canonical 0.02/0.20 parameters were tuned by Wilder on 1970s
  commodity data.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  `strategy.summarize` and `quantlab.analytics.mean_tstat_hac`.
- **Sharpe with robust SE.** Lo (2002), *The Statistics of Sharpe Ratios* (Financial
  Analysts Journal) — `quantlab.analytics.sharpe_with_se`.
- **Block bootstrap CI.** Politis & Romano (1994), *The Stationary Bootstrap* (JASA) —
  `quantlab.stats.sharpe_ci_bootstrap`.
- **ATR (Wilder's RMA).** Wilder (1978) — the risk unit R for the symmetric barriers;
  `strategy.atr` uses EWM(com=n-1), the exact Wilder formula.

## Related desk studies

- **[Study 72 — Loaded-Dice](../../72-loaded-dice/)**: SMA(5/10) crossover on 5-minute
  bars — the intraday member of the "does a trend signal beat a coin?" family; verdict
  Signal NONE, Mirage.
- **[Study 78 — Crossed-Wires](../../78-crossed-wires/)**: MACD crossover on daily bars
  — same lagging-indicator family on daily timeframe.
- **[Study 106 — Supertrend](../../106-supertrend/)**: ATR-band trend indicator on daily
  bars — the closest relative to Parabolic SAR (also Wilder-family); verdict WEAK/FRAGILE.
- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: the 50/200 golden cross — the
  slow, widely-cited moving-average trend signal; daily bars.
