# References & literature map — Study 72 (Loaded-Dice)

## The claim under test

- **The folk recipe.** A staple of day-trading forums, YouTube, and "scalping bot" tutorials:
  *"On the 5-minute chart, when the fast SMA (5) crosses the slow SMA (10), enter in the direction
  of the cross, take a few dollars of profit, and repeat all day. It's basically a coin flip,
  but the moving-average cross tips the odds in your favour."* There is no canonical paper — it is
  oral tradition — so we steelman it as the sharpest testable version: *the SMA(5/10) crossover
  direction carries enough very-short-term momentum to beat a random-direction entry, net of
  costs, on intraday bars.* The recipe is, by its own framing, an attempt to **load the dice** on
  a near-random bet — which is exactly the hypothesis we measure against an actual fair die.

## Why the steelman is *almost* coherent — the real effect it leans on

- **Weak-form efficiency / the random walk.** Fama (1970), *Efficient Capital Markets: A Review of
  Theory and Empirical Work* (Journal of Finance) — daily index returns carry negligible
  exploitable serial dependence. The recipe concedes this ("basically a coin flip") and bets the
  *intraday* micro-trend is the exception.
- **Intraday return continuation — small and horizon-specific.** Heston, Korajczyk & Sadka (2010),
  *Intraday Patterns in the Cross-section of Stock Returns* (Journal of Finance), document periodic
  intraday continuation; Gao, Han, Li & Zhou (2018), *Market Intraday Momentum* (Journal of
  Financial Economics), find the first half-hour predicts the last. These are *specific windows*,
  not "every 5-minute crossover" — and our synthetic positive control shows the cross only pays
  when such persistence actually exists in the tape.
- **At the scalp horizon, reversion usually wins.** Lo & MacKinlay (1988), *Stock Market Prices Do
  Not Follow Random Walks* (Review of Financial Studies), and the bid-ask "bounce" of Roll (1984),
  *A Simple Implicit Measure of the Effective Bid-Ask Spread* (Journal of Finance), imply that at
  one- to five-minute frequency short-term **mean reversion and microstructure noise**, not
  momentum, dominate — so a trend-following cross is often on the *wrong* side, which is what the
  real tape shows (cross gross *t* = −1.12).

## The two traps this study is really about

- **Moving-average crossovers as technical signals.** Brock, Lakonishok & LeBaron (1992), *Simple
  Technical Trading Rules and the Stochastic Properties of Stock Returns* (Journal of Finance),
  reported MA rules with predictive value — but Park & Irwin (2007), *What Do We Know About the
  Profitability of Technical Analysis?* (Journal of Economic Surveys), document how much of that
  evaporates out of sample and after data-snooping and cost adjustments. Our desk's
  [Study 21 — Fools-Gold](../../21-fools-gold/) is the daily-bar 50/200 version of the same family.
- **The high-win-rate / negative-skew illusion ("picking up pennies in front of a steamroller").**
  A small take-profit with a far (or absent) stop manufactures a 90%+ win-rate whose expectancy is
  still ≈ 0 and whose P&L skew is sharply negative — the classic "risk of ruin" payoff. Taleb
  (2004), *Fooled by Randomness*, and the gambler's-ruin literature (Feller, *An Introduction to
  Probability Theory*, Vol. 1) are the canonical framings; we reproduce it mechanically (win-rate
  93.5%, skew −8.0, *t* = 1.35).
- **Turnover kills high-frequency edges.** Novy-Marx & Velikov (2016), *A Taxonomy of Anomalies and
  Their Trading Costs* (Review of Financial Studies) — at ~11 trades/day the break-even cost a
  strategy must clear is brutal, and a coin-flip gross edge clears nothing.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.summarize`](../loaded_dice/strategy.py) and [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Sharpe with robust SE / annualisation.** Lo (2002), *The Statistics of Sharpe Ratios*
  (Financial Analysts Journal) — [`quantlab.analytics.sharpe_with_se`](../../../quantlab/analytics.py).
- **Block bootstrap CI.** Politis & Romano (1994), *The Stationary Bootstrap* (JASA) —
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).
- **Average true range.** Wilder (1978), *New Concepts in Technical Trading Systems* — the risk
  unit R for the symmetric barriers, [`strategy.atr`](../loaded_dice/strategy.py).
- **Reproducibility stamp.** [`quantlab/repro.py`](../../../quantlab/repro.py) — the as-of freeze
  and content fingerprint each headline run carries.

## Data sources used here

- **Yahoo! Finance intraday bars** (via `yfinance`), 5-minute fidelity across eight liquid tapes
  (SPY, QQQ, IWM, AAPL, TSLA, NVDA, ES=F, NQ=F). Yahoo caps sub-hourly history at ~60 calendar
  days, so this is a **low-power-by-construction** tape; the window is a rolling span ending
  ~now, and every headline is pinned with an `as_of` date and a per-tape content fingerprint
  (see [`docs/results.md`](results.md)). The offline reproducible core and the test-suite run on
  the deterministic [`data.synthetic_5m`](../loaded_dice/data.py) generator, never the network.

## Related desk studies

- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: the daily-bar 50/200 golden cross — the
  same moving-average-crossover family, one timeframe up.
- **[Study 07 — Coiled-Spring](../../07-coiled-spring/)** and
  **[Study 17 — Glass-Ceiling](../../17-glass-ceiling/)**: trend/breakout entries tested with
  barrier exits and a random baseline — the same machinery.
- **[Study 32 — Rip-Tide](../../32-rip-tide/)**: short-term reversal — the *opposite* sign of this
  rule, and a reminder that at short horizons reversion, not momentum, is the live effect.
- **[Study 22 — Crystal-Ball](../../22-crystal-ball/)** and
  **[Study 02 — Falling-Knife](../../02-falling-knife/)**: the desk's other "is this better than a
  coin?" teardowns — the random-baseline discipline this study is built around.
