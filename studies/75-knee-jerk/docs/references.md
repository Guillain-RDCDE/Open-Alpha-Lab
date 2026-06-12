# References & literature map — Study 75 (Knee-Jerk)

## The claim under test

- **The Connors RSI(2) recipe.** Connors, L. & Alvarez, C. (2008/2009), *Short Term Trading
  Strategies That Work* (TradingMarkets Publishing). The canonical statement: *"When the
  2-period RSI falls below 10 on a liquid instrument that is above its 200-day moving
  average, buy at the next open; sell when RSI(2) closes above 60."* Connors documented
  this on US equities across multi-decade backtests and claimed it was one of the most
  robust short-term patterns he had found. This is a testable hypothesis — the RSI(2)
  threshold carries directional information at a 1–10 day horizon — so we test it
  directly against a random-entry control.

## The real effect the recipe leans on — short-term mean reversion

- **Short-term equity mean reversion.** Jegadeesh (1990), *Evidence of Predictable Behavior
  of Security Returns* (Journal of Finance) — documents significant reversal over 1-month
  horizons. Lehmann (1990), *Fads, Martingales, and Market Efficiency* (Quarterly Journal
  of Economics) — finds contrarian profits at short horizons, attributed partly to
  microstructure and partly to genuine overreaction. Both papers predate the Connors recipe
  and supply the theoretical foundation: oversold stocks bounce.
- **The bid-ask bounce.** Roll (1984), *A Simple Implicit Measure of the Effective Bid-Ask
  Spread* (Journal of Finance) — at the tick level, a negative serial correlation in price
  changes is partly mechanical (trades alternating between bid and ask). A 2-period RSI on
  daily bars is less contaminated than intraday, but a fraction of the reversion is still
  microstructure.
- **Overreaction and anchoring.** De Bondt & Thaler (1985), *Does the Stock Market
  Overreact?* (Journal of Finance) — losers outperform winners at 3–5 year horizons
  (long-horizon reversion). Connors' system operates at a much shorter horizon but
  harvests the same qualitative mechanism: extreme short-term losers get bought.

## Why it decays — publication and arbitrage

- **Post-publication decay of technical anomalies.** McLean & Pontiff (2016), *Does
  Academic Research Destroy Stock Return Predictability?* (Journal of Finance) — document
  that anomalies decay substantially after publication. Our pre/post-2009 split (−35% on
  SPY) is a direct observation of this phenomenon in action: the edge shrinks but does not
  disappear, consistent with McLean-Pontiff's "partial arbitrage" finding.
- **Crowding in retail-accessible strategies.** The RSI(2) system requires no special data,
  no leverage and no HFT infrastructure — it is implementable by any retail participant
  with a brokerage account. Post-2009 widespread adoption (algorithmic retail platforms,
  "mean-reversion bots") compresses the edge: when everyone buys the oversold dip, the dip
  recovers faster and the RSI(2) buys later in the reversal.

## The 200-SMA trend filter — does it help?

- **Trend filters in mean-reversion strategies.** The conventional wisdom (amplified by
  Connors) says to only trade *with the long-term trend*: buy RSI(2) oversold *above* the
  200-day moving average. Our test contradicts this for SPY: the filter *reduces* performance
  (from t=+5.15 to t=+4.22 all-history, and undermines post-2009 returns). The reason is
  structural: the strongest RSI(2) oversold readings on SPY occur during corrections, which
  tend to push prices *below* the 200-SMA — exactly when the filter turns off. Covel (2017),
  *Trend Following* (Wiley), notes the general tension between mean-reversion and
  trend-following filters; which wins depends on the instrument and horizon.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.summarize`](../knee_jerk/strategy.py).
- **RSI indicator.** Wilder (1978), *New Concepts in Technical Trading Systems* (Trend
  Research) — the 2-period variant used here was popularised by Connors but the original
  formulation is Wilder's. The Wilder smoothing (EWM with α=1/n) matches the canonical
  implementation.
- **Forward-return test with control.** The design mirrors the desk's barrier-backtest
  studies (e.g., Study 72 — Loaded-Dice, Study 19 — Rubber-Band); here the hold is
  RSI-exit-or-max-hold rather than a price barrier, but the random-direction control
  discipline is identical.

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), adjusted-close, long history (back to
  1993 for SPY, 1999 for QQQ). Basket: SPY, QQQ, AAPL, MSFT, JPM. Pre-publication split
  at 2009-01-01. All headline numbers are pinned with an as-of date and content
  fingerprint (see [`docs/results.md`](results.md)). The offline reproducible core and
  test-suite run on the deterministic [`data.synthetic_daily`](../knee_jerk/data.py)
  generator, never the network.

## Related desk studies

- **[Study 19 — Rubber-Band](../../19-rubber-band/)**: IBS (Internal Bar Strength) is
  a related daily mean-reversion signal — buys when price closes near the day's low.
  Real signal, same "real but costs bite" tradability verdict. RSI(2) is the RSI-formulation
  cousin of IBS.
- **[Study 72 — Loaded-Dice](../../72-loaded-dice/)**: the 5-minute SMA(5/10) crossover
  scalp — the opposite of RSI(2) (momentum-following vs mean-reversion) and at much higher
  frequency. Nothing to find at intraday resolution; here the daily mean-reversion signal
  is clearly real.
- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: the 50/200 golden cross — the same
  200-SMA filter that Connors recommends, tested alone. Weak standalone signal; our result
  here shows it *hurts* when combined with RSI(2).
- **[Study 24 — Stampede](../../24-stampede/)**: cross-sectional momentum on the same
  large-cap universe. Momentum and mean-reversion are natural contrasts at different
  horizons; Stampede (12-month momentum) and Knee-Jerk (2-day RSI reversion) operate on
  orthogonal timescales.
