# References & literature map — Study 178 (CCI)

## The claim under test

- **The folk recipe.** Donald Lambert introduced the Commodity Channel Index in 1980
  (*Commodities Magazine*, October 1980: "Commodity Channel Index: Tool for Trading Cyclic
  Trends"). The original application was commodity futures, where Lambert observed that many
  commodity prices cycle through overbought and oversold conditions. The rule: when CCI
  crosses above +100 (overbought), go short; when it crosses below −100 (oversold), go long.
  The recipe is sold in modern retail communities as a universal oscillator for equities too
  — "CCI > 100 is a sell signal, CCI < −100 is a buy signal." We steelman this as: *CCI
  extremes on daily equity bars carry directional information, measured net of costs, that
  exceeds a random-direction entry on identical timing.*

## Why the steelman is *almost* coherent — the real effect it leans on

- **Lambert's original commodity context.** Lambert (1980) observed that commodity prices
  do cycle — crop cycles, seasonal demand, inventory build-and-flush — so an oscillator that
  measures distance from a moving average *does* have a natural habitat. The 0.015 constant
  was calibrated so ~70–80% of readings fall inside [−100, +100] on commodity futures.
  Equities, which trend on a structural upward path and are governed by growth expectations
  rather than physical cycles, are a different animal.
- **Contrarian / mean-reversion effects.** DeBondt & Thaler (1985), *"Does the Stock Market
  Overreact?"* (Journal of Finance), document long-horizon return reversal — but over
  3–5-year windows, not 5-day ones. Jegadeesh (1990), *"Evidence of Predictable Behavior of
  Security Returns"* (Journal of Finance), documents one-month reversal at the individual-
  stock level. These are the effects a daily oscillator might proxy, but the CCI's 5–20-day
  hold window is in an ambiguous zone between momentum-dominated and reversal-dominated.
- **Momentum dominates at intermediate horizons.** Jegadeesh & Titman (1993),
  *"Returns to Buying Winners and Selling Losers"* (Journal of Finance), document that the
  3–12-month window is momentum-dominated. At shorter windows the evidence is mixed.  The
  real tape here (hold=5–10 days) shows negative results consistent with momentum continuing
  after an "oversold" breach.

## The failure mode exposed

- **CCI as a momentum-following signal in disguise.** On trending equity instruments
  (TSLA, NVDA in this study), a CCI breach below −100 often reflects a genuine downtrend
  that continues — buying the "oversold" level adds losses. Park & Irwin (2007), *"What Do
  We Know About the Profitability of Technical Analysis?"* (Journal of Economic Surveys),
  document how technical oscillator results depend critically on the asset class, time
  period, and whether momentum or reversion dominates.
- **The win-rate is not the expectancy.** The breach framing shows a 45.1% win-rate — below
  50%, the opposite of what practitioners claim. This is not the "trick" of exit asymmetry
  (as in Study 72); here the *direction* is genuinely wrong. Consistent with Fama's (1970)
  weak-form efficiency for large liquid equities at short horizons.
- **Data-snooping and calibration.** Lambert calibrated the ±100 thresholds on 1970s
  commodity futures. Applying them to 2016–2026 US equities is an out-of-sample
  generalisation that the data do not support. Brock, Lakonishok & LeBaron (1992),
  *"Simple Technical Trading Rules and the Stochastic Properties of Stock Returns"*
  (Journal of Finance), and Sullivan, Timmermann & White (1999), *"Data-Snooping, Technical
  Trading Rule Performance, and the Bootstrap"* (Journal of Finance), document how much of
  the apparent success of technical rules disappears out of sample.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *"A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix"* (Econometrica) —
  [`strategy.summarize`](../cci/strategy.py) and [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Block bootstrap CI.** Politis & Romano (1994), *"The Stationary Bootstrap"* (JASA) —
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).
- **Reproducibility stamp.** [`quantlab/repro.py`](../../../quantlab/repro.py) — the as-of
  freeze and content fingerprint each headline run carries.

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), 10-year window (2016-2026) across six
  liquid tapes (SPY, QQQ, IWM, AAPL, TSLA, NVDA). The offline reproducible core and the
  test-suite run on the deterministic [`data.synthetic_daily`](../cci/data.py) generator,
  never the network. Each headline is pinned with an as-of date and a per-tape content
  fingerprint (see [`docs/results.md`](results.md)).

## Related desk studies

- **[Study 72 — Loaded-Dice](../../72-loaded-dice/)**: the 5-minute SMA(5/10) crossover —
  the same "does a technical rule beat a coin?" question at intraday fidelity.
- **[Study 127 — Williams-R](../../127-williams-r/)**: another normalised oscillator in the
  same overbought/oversold family, also found to carry no exploitable edge on daily equity bars.
- **[Study 106 — Supertrend](../../106-supertrend/)**: a trend-following technical rule, same
  infrastructure — different family (momentum) but same honest treatment.
- **[Study 104 — Bollinger-Reversion](../../104-bollinger-reversion/)**: Bollinger Band mean-
  reversion, the Bollinger counterpart to CCI's oscillator framing; similar result.
- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: the daily 50/200 golden cross — same
  family of "lagging indicator" teardowns.
