# References & literature map — Study 87 (Center-Line)

## The claim under test

- **The folk recipe.** A staple of day-trading communities, prop-trading handbooks, and
  "smart money" tutorials: *"Price always returns to the session VWAP — when it's stretched
  1–2 ATR above or below, fade it back. The VWAP is the centre of gravity; gravity always
  wins."* There is no canonical paper — it is practitioner oral tradition — so we steelman
  it as the sharpest testable version: *the distance of price from the running session VWAP,
  at 5-minute resolution, contains enough directional information to beat a random-direction
  entry, net of costs.* The recipe is, by its own framing, an attempt to profit from a
  predictable **gravitational pull** back to the session anchor — which is exactly the
  hypothesis we measure against an actual fair coin.

## Why the steelman is almost coherent — the real effect it leans on

- **VWAP as institutional anchor.** Berkowitz, Logue & Noser (1988), *The Total Cost of
  Transactions on the NYSE* (Journal of Finance), and later the development of VWAP
  execution benchmarks, establish VWAP as the dominant institutional execution target. Large
  orders are typically scheduled to track VWAP, which creates a genuine mechanical pull: new
  institutional flow enters near the VWAP throughout the day, creating real short-term demand
  near the price centre. This is the microstructure mechanism the recipe leans on.
- **Intraday mean-reversion at short horizons.** Roll (1984), *A Simple Implicit Measure of
  the Effective Bid-Ask Spread* (Journal of Finance), shows bid-ask bounce produces negative
  one-lag autocorrelation in transaction prices at short intervals. Hasbrouck (1993),
  *Assessing the Quality of a Securities Market* (Journal of Finance), and the market
  microstructure literature broadly document price reversion at sub-minute horizons — but
  the effect decays rapidly and is largely absorbed by the spread, not a windfall for
  momentum traders.
- **VWAP is a moving average by construction.** The running session VWAP is a volume-weighted
  cumulative average of all prices traded that session. Mean-reversion *toward* VWAP is partly
  a mathematical identity (extreme moves from any running average tend to revert toward it in
  expectation, by the properties of running averages), not necessarily a forecasting signal.
  This is the core confound the study is designed to expose.

## Why the effect is likely to fail as a trading signal

- **The ATR threshold fires too often.** A 1-ATR deviation from the VWAP is a routine
  intraday move, not a true overshoot. On the 60-day window, the average instrument fires
  ~49 signals per day — that is ~63% of all RTH bars. A signal that fires constantly is
  not a selective filter; it is a description of the average bar.
- **Weak-form efficiency at the 5-minute horizon.** Fama (1970), *Efficient Capital Markets:
  A Review of Theory and Empirical Work* (Journal of Finance) — competition among statistical
  arbitrageurs rapidly eliminates mean-reversion patterns at the 5-minute scale that are
  widely known and widely traded. Chordia, Roll & Subrahmanyam (2008), *Liquidity and Market
  Efficiency* (Journal of Financial Economics), document that intraday return autocorrelations
  are near zero for liquid US equities after accounting for microstructure noise.
- **Turnover cost dominates.** Novy-Marx & Velikov (2016), *A Taxonomy of Anomalies and Their
  Trading Costs* (Review of Financial Studies) — at ~49 trades/day the annual break-even cost
  is essentially undefined; even a near-zero per-trade edge is destroyed by round-trip friction.
  The strategy requires a gross per-trade edge substantially above zero *before* costs; we
  measure −0.35 bps gross.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.summarize`](../center_line/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Block bootstrap CI.** Politis & Romano (1994), *The Stationary Bootstrap* (JASA) —
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).
- **Average true range.** Wilder (1978), *New Concepts in Technical Trading Systems* — the
  risk unit R for the symmetric barriers and the deviation threshold.
  [`strategy.atr`](../center_line/strategy.py).
- **Random-direction control.** The desk's standard baseline (equal-exposure, unordered
  direction) — ensures any measured edge is attributable to the direction signal, not to the
  entry timing or the payoff structure.
- **Reproducibility stamp.** [`quantlab/repro.py`](../../../quantlab/repro.py) — the as-of
  freeze and content fingerprint each headline run carries.

## Data sources used here

- **Yahoo! Finance intraday bars** (via `yfinance`), 5-minute fidelity across eight liquid
  tapes (SPY, QQQ, IWM, AAPL, TSLA, NVDA, ES=F, NQ=F). Yahoo caps sub-hourly history at
  ~60 calendar days, so this is a **low-power-by-construction** tape; the window is a rolling
  span ending ~now, and every headline is pinned with an `as_of` date and a per-tape content
  fingerprint (see [`docs/results.md`](results.md)). The offline reproducible core and the
  test-suite run on the deterministic [`data.synthetic_5m`](../center_line/data.py) generator,
  never the network.

## Related desk studies

- **[Study 72 — Loaded-Dice](../../72-loaded-dice/)**: the SMA(5/10) crossover on the same
  five-minute tape — the same infrastructure, a different technical signal, the same verdict
  (NONE / MIRAGE). Both studies confirm that short-horizon technical rules on liquid
  instruments earn no detectable edge once tested honestly.
- **[Study 13 — Crimson-Hour](../../13-crimson-hour/)**: intraday time-of-day patterns —
  tests whether specific *windows* of the session (not VWAP deviation) carry exploitable
  structure, including the opening-range and closing-auction effects.
- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: the daily 50/200 golden cross — same
  moving-average family as the VWAP, one timeframe up, same family of null results.
- **[Study 67 — Fed-Drift](../../67-fed-drift/)**: event-driven intraday drift around FOMC
  announcements — a study where a genuine *specific* intraday window (the pre-FOMC hours)
  is tested for real signal, unlike the broad "any deviation from VWAP" claim here.
