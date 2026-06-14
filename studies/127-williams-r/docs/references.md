# References & literature map — Study 127 (Williams-R)

## The claim under test

- **The folk recipe.** Larry Williams popularised %R in his 1979 book *How I Made One
  Million Dollars Last Year Trading Commodities* and subsequent work. The rule is ubiquitous
  in retail trading platforms (TradingView, ThinkOrSwim, TradeStation): on the daily chart,
  when %R dips below −80 ("oversold"), buy the next open expecting a bounce; when %R rises
  above −20 ("overbought"), short the next open expecting a pullback. The claim is that the
  14-period position of the close within the high-low range identifies *exhausted moves* and
  predicts a near-term reversal — a directional forecast that should outperform a random
  coin on the same entry dates.

## Why the steelman is almost coherent

- **Short-term mean-reversion in equity prices.** Lo & MacKinlay (1988), *Stock Market
  Prices Do Not Follow Random Walks* (Review of Financial Studies), document weekly
  return reversals in individual stocks — the foundation of the "price overshoots and
  snaps back" story. Jegadeesh (1990), *Evidence of Predictable Behavior of Security
  Returns* (Journal of Finance), reinforces one-month reversals. The %R bet is a daily
  version of the same idea.
- **The stochastic oscillator — the same mathematics.** Williams %R is algebraically
  equivalent to (1 − Stochastic %K), so all evidence for the stochastic oscillator bears
  on %R. Lane (1984) introduced Stochastic %K on the same high-low-close logic; empirical
  evidence for it is mixed to weak. See also Study 107 (Stochastic-Oscillator) at this
  desk for a parallel test that finds the same NONE/MIRAGE result.
- **Oscillator predictability — the weak evidence base.** Pruitt & White (1988), *The
  CRISMA Trading System: Who Really Profits?* (Journal of Portfolio Management), find
  oscillator-based rules profitable in-sample but not robustly out-of-sample. Brock,
  Lakonishok & LeBaron (1992), *Simple Technical Trading Rules and the Stochastic
  Properties of Stock Returns* (Journal of Finance), show some predictability from
  technical rules in the pre-1990 era — but Park & Irwin (2007), *What Do We Know About
  the Profitability of Technical Analysis?* (Journal of Economic Surveys), document how
  much of it evaporates after data-snooping adjustments and cost corrections. Bajgrowicz &
  Scaillet (2012), *Technical Trading Revisited* (Journal of Financial Economics), apply
  a rigorous multiple-testing correction and find no surviving rules in recent data.

## The two traps this study is really about

- **Oscillator illusion: the zone entry is usually momentum, not reversal.** When %R
  enters the oversold zone (<−80), the price has *just fallen*; holding direction of the
  last move for a day is equivalent to a short-term momentum bet. The data confirm this:
  the 1-day hold shows +2.9 bps (minor positive but noisy), and the 5-day and 10-day
  holds are increasingly negative — price continues in the direction of the move rather
  than reversing. The oscillator triggers too early.
- **The cross-back framing enters too late.** Waiting for %R to exit the oversold zone
  (the "confirmed reversal") means entering *after* the bounce has largely occurred —
  which is why the cross-back framing shows a statistically significant *negative* result
  (t = −2.14 at hold=5). Both timing choices lose.
- **Turnover arithmetic.** ~35 signals/ticker/year means costs are not the primary killer
  here (unlike purely intraday strategies) — the real problem is that the gross expectancy
  is negative across all hold periods tested.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.summarize`](../williams_r/strategy.py).
- **Random-direction control.** The primary comparison: same entry dates, direction drawn
  i.i.d. fair coin — the desk's standard "is this better than a coin?" baseline, used in
  Studies 72, 78, 106, 107, and 127.
- **Fixed-horizon forward return.** Entry at *t+1* open, exit at close of bar *t+N*. The
  simplest look-ahead-free measurement of directional skill.

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), ten years of history across six tickers:
  SPY, QQQ, IWM, AAPL, TSLA, NVDA. Window 2016-06-13 → 2026-06-12, ~2,515 bars per
  instrument. The offline reproducible core and test-suite run on the deterministic
  [`data.synthetic_daily`](../williams_r/data.py) generator, never the network.

## Related desk studies

- **[Study 107 — Stochastic-Oscillator](../../107-stochastic-oscillator/)**: %K and %D
  — the algebraically equivalent Lane stochastic — tested with the same protocol; same
  NONE/MIRAGE result.
- **[Study 72 — Loaded-Dice](../../72-loaded-dice/)**: the SMA(5/10) 5-minute crossover
  scalp — the same "coin with an edge?" framing applied to trend-following vs. the
  coin; NONE/MIRAGE there too.
- **[Study 104 — Bollinger-Reversion](../../104-bollinger-reversion/)**: mean-reversion
  entry at Bollinger Band extremes — a different normalisation of the same "price at the
  edge of its range" idea.
- **[Study 106 — Supertrend](../../106-supertrend/)**: ATR-band trend-following — in
  the same technical-indicator family, but using the direction instead of the level.
