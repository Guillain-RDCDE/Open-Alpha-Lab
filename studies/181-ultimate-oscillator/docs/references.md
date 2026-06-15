# References & literature map — Study 181 (Ultimate-Oscillator)

## The claim under test

**The folk recipe.** Larry Williams (1985) introduced the Ultimate Oscillator as an improvement
over single-period momentum oscillators (RSI, Stochastic) by weighting three look-back periods
(7, 14, 28 bars).  The published trading rules:

- **Oversold (< 30):** buy at the next bar's open; expect a short-term bounce.
- **Overbought (> 70):** short at the next bar's open; expect a short-term pullback.
- **Bullish divergence:** price makes a new low but UO makes a higher low → buy signal.
- **Bearish divergence:** price makes a new high but UO makes a lower high → sell signal.

We test both framings honestly: are they directionally useful *beyond what a fair coin on the
same entry bars would earn*?

## The originating work

- **Williams, L. (1985).** *The Ultimate Oscillator.* Technical Analysis of Stocks & Commodities,
  3(4), 140–141.  The original description of the weighted three-period oscillator and the
  divergence-based entry/exit rules.  Williams claimed the multi-period weighting reduced false
  signals; we test that claim directly.

## Why the claim is plausible — the real effect it leans on

- **Short-term mean reversion in equity returns.**  De Bondt & Thaler (1985), *Does the Stock
  Market Overreact?* (Journal of Finance), document return reversals at 3–5-year horizons.
  Jegadeesh (1990), *Evidence of Predictable Behavior of Security Returns* (Journal of Finance),
  shows significant 1-month reversal at the monthly scale.  At the daily scale the effect is
  smaller but documented: Lo & MacKinlay (1988), *Stock Market Prices Do Not Follow Random Walks*
  (Review of Financial Studies), show positive first-order autocorrelation in weekly index returns
  that is consistent with mean reversion at intermediate frequencies.

- **Buying-pressure decomposition as a microstructure measure.**  The UO's BP/TR ratio
  (close - min(low, prev_close)) / (max(high, prev_close) - min(low, prev_close)) is conceptually
  close to Stochastic %K and to Ease of Movement — all are within-range normalised measures of
  net buying pressure.  Roll (1984), *A Simple Implicit Measure of the Effective Bid-Ask Spread*
  (Journal of Finance), and Amihud (2002), *Illiquidity and Stock Returns* (Journal of Financial
  Markets), ground the intuition that selling pressure episodes (low UO) carry predictable
  short-term reversal as liquidity providers rebalance.

## Why the claim is weak on our tape

- **Low signal frequency.**  The UO threshold rule fires ≈ 7 times per ticker per year — about
  once every 7–8 weeks.  With only 5 tickers × 20 years = 681 pooled entries, the HAC t-stat
  of +1.50 is insufficient for certification (|t| ≥ 2.0 bar).  Brock, Lakonishok & LeBaron
  (1992), *Simple Technical Trading Rules and the Stochastic Properties of Stock Returns*
  (Journal of Finance), show that even statistically significant-looking technical rules often
  fail out-of-sample; our 681-trade sample is too thin to rule out luck.

- **Multiple-comparisons problem in hold-period selection.**  The "best" hold period (1 day,
  t = +2.14) is selected from five candidates.  Harvey, Liu & Zhu (2016), *...and the
  Cross-Section of Expected Returns* (Review of Financial Studies), argue that any anomaly
  discovered through a parameter search requires t > 3 to be believable.  Our best t = 2.14 falls
  well short.

- **Divergence rule fails.**  The divergence framing (n = 5,336) earns −7.08 bps/trade at
  t = −1.80 — actively negative.  This is consistent with Lui & Mole (1998), *The Use of
  Fundamental and Technical Analyses by Foreign Exchange Dealers* (Journal of International Money
  and Finance), who find divergence-based rules are among the weakest performers out-of-sample.

## The desk's related studies

- **[Study 127 — Williams-R](../../127-williams-r/):** the parent oscillator — Williams' %R (same
  author, same oversold/overbought framing) tested on the same basket with the same protocol.
  Both studies share the mean-reversion thesis; Williams-R's verdict is similarly weak/none.

- **[Study 106 — Supertrend](../../106-supertrend/):** another daily-bar trend indicator tested
  via fixed-horizon backtest and random-direction control — the same engine and inference bar.

- **[Study 21 — Fools-Gold](../../21-fools-gold/):** the daily 50/200 moving-average cross — the
  trend-following side of the same technical-indicator family.

- **[Study 72 — Loaded-Dice](../../72-loaded-dice/):** the 5-minute SMA(5/10) crossover scalp —
  the same honest-bet-vs-coin methodology applied at intraday resolution.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.summarize`](../ultimate_oscillator/strategy.py).
- **Bonferroni correction.** Shaffer (1995), *Multiple Hypothesis Testing* (Annual Review of
  Psychology) — applied across the 5-point hold-period sweep.
- **Reproducibility stamp.** Content fingerprint per tape in [`docs/results.md`](results.md).

## Data sources used here

- **Yahoo Finance daily bars** (via `yfinance`), 20-year window, five liquid ETFs (SPY, QQQ, IWM,
  GLD, TLT).  Cache-only by default; the offline core and test-suite run entirely on the
  deterministic synthetic tape in [`data.synthetic_daily`](../ultimate_oscillator/data.py).
