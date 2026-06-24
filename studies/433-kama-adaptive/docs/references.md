# References & literature map — Study 433 (Kaufman Adaptive MA)

## The claim under test

- **The folk recipe.** Perry J. Kaufman introduced the Adaptive Moving Average (AMA/KAMA)
  in *Smarter Trading: Improving Performance in Changing Markets* (McGraw-Hill, 1995), and
  developed it further in *Trading Systems and Methods* (Wiley, multiple editions). The pitch:
  an ordinary moving average uses a *fixed* lookback, so it lags badly in trends and whipsaws
  in chop. KAMA fixes this by scaling its EMA smoothing constant with an **Efficiency Ratio**
  (net travel ÷ total path), tightening to a fast EMA in clean trends and freezing to a slow
  EMA in noise. The marketing claim sold across trading platforms and forums: *a price-cross
  timing rule on KAMA beats the same rule on a fixed SMA — fewer whipsaws, faster entries.*
  We steelman this as: *the KAMA-cross book's net excess Sharpe exceeds the matched SMA-cross
  book's, with the KAMA − SMA daily out-performance significant on the real tape.*

## Why the steelman is *almost* coherent — the real effect it leans on

- **The Efficiency Ratio is a real concept.** Kaufman's ER (a.k.a. the "fractal efficiency"
  of a path) genuinely separates directional moves from chop — it is 1 for a straight line
  and ~0 for a sawtooth. The intuition that you'd want a faster filter in trends and a slower
  one in chop is sound *in principle*.
- **Regime dependence of MA rules.** Brock, Lakonishok & LeBaron (1992), *"Simple Technical
  Trading Rules and the Stochastic Properties of Stock Returns"* (Journal of Finance),
  document that moving-average rules' performance is highly regime-dependent — exactly the
  weakness KAMA claims to patch.
- **Trend-following has a real (thin) premium.** Moskowitz, Ooi & Pedersen (2012), *"Time
  Series Momentum"* (Journal of Financial Economics), and Faber (2007), *"A Quantitative
  Approach to Tactical Asset Allocation"*, show that a simple trend filter can cut drawdowns —
  which is why the *SMA* book here beats KAMA *and* has a far shallower drawdown than holding.

## The failure mode exposed

- **Adaptation adds churn, not edge.** On the real tape KAMA's turnover is *higher* than the
  SMA's (1269 vs 765 on SPY), and its net Sharpe is *lower* (+0.20 vs +0.53). The squared
  smoothing constant keeps the filter creeping across the price even when ER is low, so the
  promised whipsaw avoidance does not materialise; costs make the gap worse. This is the
  classic *over-parameterised technical rule* failure documented by Park & Irwin (2007),
  *"What Do We Know About the Profitability of Technical Analysis?"* (Journal of Economic
  Surveys) — extra knobs rarely beat the simple version out of sample.
- **Data-snooping / out-of-sample generalisation.** Kaufman tuned KAMA's defaults (ER=10,
  fast=2, slow=30) on his own data decades ago. Sullivan, Timmermann & White (1999),
  *"Data-Snooping, Technical Trading Rule Performance, and the Bootstrap"* (Journal of
  Finance), show how much of a technical rule's apparent edge evaporates once you correct for
  the search over its parameters — and our robustness grid finds *no* parameter where KAMA
  beats the SMA.
- **The benchmark matters.** Comparing KAMA to *holding* alone would have flattered it (any
  trend filter cuts drawdown); the honest comparison is KAMA vs the *fixed* filter it claims
  to improve, and there it loses.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *"A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix"* (Econometrica) —
  [`strategy.hac_tstat`](../kama_adaptive/strategy.py).
- **Permutation / label-shuffle placebo.** The position-shuffle null follows the
  data-mining-control logic of Sullivan-Timmermann-White (1999) and the desk's
  research-method demos (Studies 343–350).
- **Reproducibility stamp.** As-of freeze + content fingerprint
  ([`data.fingerprint`](../kama_adaptive/data.py)); the partial current month is dropped.

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`, `auto_adjust=True` → split/dividend
  adjusted, total-return-ish closes), full history to **2026-05-31** across six liquid tapes
  (SPY, QQQ, IWM, EFA, GLD, AAPL). The reproducible core and the offline synthetic positive
  control run on [`data.synthetic_panel`](../kama_adaptive/data.py), never the network. Each
  headline is pinned with an as-of date and a per-tape content fingerprint
  (see [`docs/results.md`](results.md)).

## Related desk studies

- **[Study 104 — Bollinger-Reversion](../../104-bollinger-reversion/)**: a fixed-window MA
  band rule — the same "does a technical filter beat the simple benchmark?" question.
- **[Study 178 — CCI](../../178-cci/)**: Lambert's oscillator, same honest "beats a coin /
  beats the simple version?" treatment on daily equity bars.
- **[Study 106 — Supertrend](../../106-supertrend/)**: an ATR-adaptive trend filter — KAMA's
  closest cousin (adaptation via volatility), tested with the same infrastructure.
- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: the 50/200 golden cross — the canonical
  fixed-MA timing teardown.
- **[Study 344 — Backtest-Overfitting](../../344-backtest-overfitting/)**: why adding knobs to
  a rule manufactures in-sample edge that dies out of sample — the mechanism behind KAMA's miss.
