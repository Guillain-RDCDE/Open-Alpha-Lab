# References & literature map — Study 422 (Elder Ray)

## The claim under test

- **The folk recipe.** Dr Alexander Elder introduced *Elder Ray* in *Trading for a Living:
  Psychology, Trading Tactics, Money Management* (Wiley, 1993; the indicator dates to his
  late-1980s teaching). It reads price as a tug of war around a 13-period EMA "consensus of
  value": **Bull Power = High − EMA13** (how far buyers push above value) and
  **Bear Power = Low − EMA13** (how far sellers push below it). Elder's rule, the *second
  screen* of his **Triple Screen** trading system: trade only with the EMA trend, and time
  the entry with the power oscillators — in an uptrend, buy when Bear Power is negative but
  *rising* (bears exhausted). We steelman this as: *the Bull/Bear Power decomposition, used
  as a long/flat timing rule, beats buy-and-hold net of costs and beats the obvious simpler
  trend filters it is built on.*

## Why the steelman is *almost* coherent — the real effect it leans on

- **Trend-following does work, sometimes.** Moskowitz, Ooi & Pedersen (2012), *"Time Series
  Momentum"* (Journal of Financial Economics), document a real, cross-asset trend premium —
  so a trend filter (which Elder Ray is, at heart) is not crazy. The question is whether the
  Bull/Bear Power *detail* adds anything over the EMA trend itself.
- **Elder's own framing is a system, not a single signal.** Elder (1993) never recommends
  Elder Ray in isolation; it is one of three screens (a longer-timeframe trend, the daily
  Elder Ray / oscillator, and an entry trigger such as the Force Index). Testing the second
  screen alone is a fair *isolation* of the indicator's marginal value, and is explicitly
  noted as such on the Signal axis.
- **Drawdown reduction is genuine.** Any "step aside when the trend turns down" rule does
  cut tail risk (here −39% vs −55% on SPY) — Faber (2007), *"A Quantitative Approach to
  Tactical Asset Allocation"* (Journal of Wealth Management), shows the 10-month SMA filter
  trades return for a milder drawdown. Elder Ray inherits that, but loses the return race.

## The failure mode exposed

- **The indicator loses to its own simplest component.** A bare EMA13 cross (Elder Ray
  *without* the power oscillators) and a plain 200-day SMA filter both beat the full rule on
  Sharpe, CAGR and HAC *t*. The Bull/Bear Power machinery *subtracts* value — a textbook
  case of a multi-part indicator failing the marginal-value test. Park & Irwin (2007),
  *"What Do We Know About the Profitability of Technical Analysis?"* (Journal of Economic
  Surveys), survey how rarely complex technical rules beat simple ones out of sample.
- **Drawdown reduction is not alpha.** Sitting in cash 54% of the time cuts the drawdown but
  also the return; on a risk-adjusted (Sharpe) basis the rule is *below* buy-and-hold. The
  "edge" is just reduced beta exposure, repriced.
- **Data-snooping / fragility.** Brock, Lakonishok & LeBaron (1992), *"Simple Technical
  Trading Rules and the Stochastic Properties of Stock Returns"* (Journal of Finance), and
  Sullivan, Timmermann & White (1999), *"Data-Snooping, Technical Trading Rule Performance,
  and the Bootstrap"* (Journal of Finance), document how much apparent technical-rule
  success evaporates once you account for the universe of rules tried. The rotation placebo
  here (*p* = 0.99) makes the same point directly: the realised timing is no better than
  random alignment.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *"A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix"* (Econometrica) —
  [`strategy.hac_tstat`](../elder_ray/strategy.py).
- **Rotation / circular-shift placebo.** A position-vs-return alignment test in the spirit
  of the stationary bootstrap of Politis & Romano (1994), *"The Stationary Bootstrap"*
  (JASA) — [`strategy.rotation_placebo`](../elder_ray/strategy.py).
- **Excess-vs-excess Sharpe & one-day execution lag.** House conventions in
  [METHODOLOGY.md](../../../METHODOLOGY.md); the lag is a single `shift` in
  [`strategy.backtest`](../elder_ray/strategy.py).
- **Reproducibility stamp.** As-of freeze + per-tape content fingerprint
  ([`data.fingerprint`](../elder_ray/data.py)); see [`docs/results.md`](results.md).

## Data sources used here

- **Yahoo! Finance daily total-return bars** (via `yfinance`, `auto_adjust=True`), full
  history through **2026-05-31**, across SPY, QQQ, IWM, EFA, GLD. The offline reproducible
  core, the synthetic positive control, and the rotation placebo run on the deterministic
  [`data.synthetic_panel`](../elder_ray/data.py) generator, never the network. Each headline
  is pinned with an as-of date and a per-tape fingerprint (see [`docs/results.md`](results.md)).

## Related desk studies

- **[Study 178 — CCI](../../178-cci/)**: Lambert's Commodity Channel Index — another
  oscillator overbought/oversold rule, also `NONE × MIRAGE` on modern equity daily bars.
- **[Study 104 — Bollinger-Reversion](../../104-bollinger-reversion/)**: Bollinger Band
  mean-reversion, the band counterpart to Elder Ray's value-band framing; same outcome.
- **[Study 106 — Supertrend](../../106-supertrend/)**: a trend-following technical rule on
  the same infrastructure — the family Elder Ray actually belongs to.
- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: the daily 50/200 golden cross — the
  classic "lagging trend filter that trades return for drawdown" teardown.
