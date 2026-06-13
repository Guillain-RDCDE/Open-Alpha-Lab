# References & literature map — Study 107 (Stochastic-Oscillator)

## The claim under test

- **The folk recipe.** George Lane's Stochastic Oscillator (1950s–1980s), popularised in
  *Trading for a Living* (Elder, 1993) and countless online tutorials: *"Compute %K as the
  position of the current close within the last 14 periods' high-low range, smooth it to
  %D(3), and trade the cross — buy when %K crosses above %D in oversold territory (both
  below 20), sell when it crosses below %D in overbought territory (both above 80). The
  cross in an extreme zone signals a momentum exhaustion and an imminent reversal."* We
  steelman it as the sharpest testable version: *the zone-filtered %K/%D crossover carries
  enough short-term mean-reversion information to beat a random-direction entry, net of
  costs, over a 5-day forward window on daily equity data.*

## The underlying effect — why the steelman is almost coherent

- **Short-term mean reversion in equity returns.** De Bondt & Thaler (1985), *Does the
  Stock Market Overreact?* (Journal of Finance) — 3–5-year mean reversion; Jegadeesh
  (1990), *Evidence of Predictable Behavior of Security Returns* (Journal of Finance) —
  monthly reversals at the 1-month horizon.  A stochastic near an extreme is, in effect,
  betting on a milder version of the same process at the weekly scale.
- **Overbought/oversold as a contrarian signal.** Wilder (1978), *New Concepts in
  Technical Trading Systems*, introduced RSI with overbought/oversold thresholds on
  similar logic; Lane's Stochastic applies the same intuition to the range-normalised
  close.  Both assume that a sustained extreme reading reflects temporary overextension
  that will self-correct — the empirical question is whether the correction is fast enough
  and reliable enough to trade.
- **In practice: mixed and context-dependent.** Brock, Lakonishok & LeBaron (1992),
  *Simple Technical Trading Rules and the Stochastic Properties of Stock Returns* (Journal
  of Finance), found in-sample predictive power for moving-average rules on DJIA but the
  stochastic was not specifically tested.  Park & Irwin (2007), *What Do We Know About the
  Profitability of Technical Analysis?* (Journal of Economic Surveys) — meta-analysis of
  95 studies: profits exist in early periods, evaporate after 1990 and especially
  post-transaction-cost on liquid markets.

## Why it likely fails on modern daily equity data

- **Secular uptrend bias.** In a 10-year bull-heavy window (2016–2026), the stochastic
  spends far more time in overbought than oversold territory for US large-caps.  This
  causes an asymmetric signal count (SPY: 33 buys vs 258 sells) and biases the pooled
  result negative through systematic short exposure to an uptrending index.  The 10-day
  hold result (t = −2.71) is a benchmark-leak artifact, not an exploitable edge.
- **Microstructure and efficient-market arguments.** Lo & MacKinlay (1988), *Stock Market
  Prices Do Not Follow Random Walks* (Review of Financial Studies) — daily returns show
  *positive* autocorrelation at short horizons (momentum), not the negative autocorrelation
  the stochastic-reversal rule bets on.  Short-term mean reversion is well-documented at
  *intraday* (bid-ask bounce, Roll 1984) and *monthly* (Jegadeesh 1990) scales, but is
  weakest at the *daily* horizon this study tests.
- **Out-of-sample decay.** Sullivan, Timmermann & White (1999), *Data-Snooping, Technical
  Trading Rule Performance, and the Bootstrap* (Journal of Finance) — rules selected from
  the same universe as Brock et al. have near-zero out-of-sample predictive power once
  data-snooping is corrected for.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.summarize`](../stochastic_oscillator/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Forward-return barrier backtest.** The fixed-horizon framing (enter at *t+1* open,
  exit at *t+hold* close) is standard in the event-study literature: Campbell, Lo & MacKinlay
  (1997), *The Econometrics of Financial Markets*, Chapter 4; no look-ahead by construction.
- **Random-direction control.** The coin-flip control is equivalent to a permutation test
  of the directional information content; see Brock et al. (1992) and the broader
  literature on technical analysis evaluation.
- **Block bootstrap CI.** Politis & Romano (1994), *The Stationary Bootstrap* (JASA) —
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), 10-year window 2016-06-13 to 2026-06-12,
  across six liquid instruments (SPY, QQQ, IWM, AAPL, TSLA, NVDA).  Prices are
  adjusted for splits and dividends (`auto_adjust=True`).  Every headline is pinned with
  an `as_of` date and a per-tape content fingerprint (see [`docs/results.md`](results.md)).
  The offline reproducible core and test-suite run on the deterministic
  [`data.synthetic_daily`](../stochastic_oscillator/data.py) generator, never the network.

## Related desk studies

- **[Study 72 — Loaded-Dice](../../72-loaded-dice/)**: the SMA(5/10) 5-minute crossover
  scalp — same "crossover as directional signal" family, intraday fidelity; also beats a
  coin? Also no.
- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: daily 50/200 golden cross — another
  moving-average crossover from the same technical-analysis tradition.
- **[Study 78 — Crossed-Wires](../../78-crossed-wires/)**: MA/indicator rule with the same
  machinery; useful domain comparison for signal quality.
- **[Study 86 — Tail-Radar](../../86-tail-radar/)**: vol-index signals as regime filters —
  a related "extreme reading → fade" idea applied to implied volatility.
- **[Study 85 — Dr-Copper](../../85-dr-copper/)**: cross-asset ratio predictors on daily
  data — the same daily-bar forward-return framing on a different signal family.
