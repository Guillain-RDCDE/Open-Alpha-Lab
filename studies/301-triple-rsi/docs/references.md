# References & literature map — Study 301 (Triple-RSI)

## The claim under test

- **The "Triple RSI" recipe.** QuantifiedStrategies.com (Oddmund Grøtte & Håkan
  Samuelsson), *Triple RSI Trading Strategy: Boost Your Win Rate to 90%*. The canonical
  statement: on SPY, buy at the close when (1) the 5-day RSI is below 30, (2) the 5-day
  RSI is down for the third day in a row, (3) the 5-day RSI three trading days ago was
  below 60, and (4) the close is above the 200-day moving average; sell at the close when
  the 5-day RSI crosses above 50. The marketed result is *"few trades but a solid 90% win
  rate."* This is a testable hypothesis — the four-condition RSI(5) stack carries
  positive-expectancy directional information at a few-day horizon — so we test it
  directly against a random-entry control, and we interrogate the headline win-rate.

## The real effect the recipe leans on — short-term mean reversion

- **Short-term equity mean reversion.** Jegadeesh (1990), *Evidence of Predictable
  Behavior of Security Returns* (Journal of Finance) — significant reversal over short
  horizons. Lehmann (1990), *Fads, Martingales, and Market Efficiency* (Quarterly Journal
  of Economics) — contrarian profits at short horizons. These supply the foundation:
  oversold instruments bounce. The Triple-RSI is one (heavily over-conditioned) way to
  select oversold entries; its real signal is the same effect harvested by Study 75.
- **The bid-ask bounce.** Roll (1984), *A Simple Implicit Measure of the Effective Bid-Ask
  Spread* (Journal of Finance) — a fraction of short-horizon reversion is mechanical
  microstructure; a 5-period RSI on daily index ETFs is largely clear of it but not
  entirely.

## The headline number — why a high win-rate is the wrong metric

- **Win-rate vs expectancy.** A high hit-rate with an asymmetric exit (take small gains
  quickly, ride losers) is the classic shape of a *negative-skew* payoff: many small wins,
  rare large losses. The win-rate can be high while the expectancy is zero or negative —
  the desk demonstrates this on a literal coin (synthetic martingale: 62% wins, negative
  mean). The relevant statistic is the mean and its HAC *t*, not the fraction of winners.
  This mirrors the exit-asymmetry illusion documented in [Study 72 — Loaded-Dice](../../72-loaded-dice/)
  (a fixed-tick small-take-profit arm printed a 93.5% win-rate at skew −8.0 on noise).
- **The Sharpe/skew trade-off.** Taleb (2004), *Fooled by Randomness* — the danger of
  strategies that "win often" and blow up rarely. The Triple-RSI is a mild version: real
  edge, but the −1.6 skew means the win-rate oversells the comfort.

## Capacity, not cost, is the binding constraint

- **Capital efficiency of sparse-signal systems.** With ~3.5 trades/yr and a ~5-day hold,
  the strategy is in the market only ~7% of the time. Even a large per-trade edge then
  compounds slowly: the annualised contribution (~+4.7%/yr) trails simple buy-and-hold
  (~+10.8%/yr). This is the standard tension for highly-conditioned entry rules — each
  extra filter raises the win-rate and the per-trade edge while shrinking the trade count
  toward statistical and practical irrelevance.
- **Over-conditioning / degrees of freedom.** Harvey, Liu & Zhu (2016), *…and the
  Cross-Section of Expected Returns* (Review of Financial Studies) — the multiple-testing
  problem. Three of the four Triple-RSI conditions are different cuts of the same RSI(5)
  series; stacking near-collinear filters is exactly the recipe for an impressive
  in-sample win-rate on a thin sample. Our pre/post-2010 split and next-open robustness
  check guard against the worst of it; the durable post-2010 *t* = +3.11 is reassuring.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.summarize`](../triple_rsi/strategy.py).
- **RSI indicator.** Wilder (1978), *New Concepts in Technical Trading Systems* (Trend
  Research). The Wilder smoothing (EWM with α = 1/n) matches the canonical implementation.
- **Forward-return test with control.** The design mirrors the desk's barrier-backtest
  studies; the random-direction control discipline is identical to Study 75 (Knee-Jerk)
  and Study 72 (Loaded-Dice).

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), adjusted-close, long history (back to
  1993 for SPY). Basket: SPY, QQQ, IWM, DIA. Pre/post split at 2010-01-01. All headline
  numbers are pinned with an as-of date and content fingerprint (see
  [`docs/results.md`](results.md)). The offline reproducible core and test-suite run on the
  deterministic [`data.synthetic_daily`](../triple_rsi/data.py) generator, never the network.

## Related desk studies

- **[Study 75 — Knee-Jerk](../../75-knee-jerk/)**: the Connors RSI(2) < 10 oversold-bounce
  system — the direct cousin. Same family (daily RSI mean-reversion), same Real signal,
  same Fragile tradability. Knee-Jerk fires ~24 trades/yr; Triple-RSI's extra conditions
  buy a higher win-rate at the cost of ~7× fewer trades.
- **[Study 72 — Loaded-Dice](../../72-loaded-dice/)**: where the win-rate illusion was
  first dissected on this desk — a fixed-take-profit exit printed a 93.5% win-rate on pure
  noise. Triple-RSI is the honest counterpart: the win-rate illusion is present, but here
  there is also a real signal underneath it.
- **[Study 19 — Rubber-Band](../../19-rubber-band/)**: IBS (Internal Bar Strength), another
  daily mean-reversion entry. Same family, same "real but costs/capacity bite" verdict.
- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: the 50/200 golden cross — the same
  200-SMA trend filter tested in isolation (weak standalone).
