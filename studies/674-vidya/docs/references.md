# References & literature map — Study 674 (VIDYA)

## The claim under test

- **The original source.** Tushar Chande, *"Adapting Moving Averages to Market
  Volatility"*, Technical Analysis of Stocks & Commodities, March 1992 — later folded
  into Chande & Kroll, *"The New Technical Trader"* (Wiley, 1994). Chande's own pitch:
  a moving average whose smoothing constant is scaled by his Chande Momentum Oscillator
  (CMO — Study 185) "automatically adjusts" — moving fast in "volatile, trending
  markets" and slow ("almost flat") in "non-trending, low-volatility markets" — so a
  crossover rule on it should beat a fixed-window SMA/EMA of the same nominal length.
  Every retail charting platform's VIDYA glossary entry since (StockCharts,
  TradingView, MetaTrader indicator packs) repeats the "volatile-trending" framing
  verbatim. We steelman it as: *a VIDYA(14, cmo=9) price-cross long/flat rule beats the
  equivalent SMA(14)/EMA(14) rule AND beats buy-and-hold, net of costs, and its own
  smoothing constant genuinely responds to volatile/trending conditions.*

## Why the steelman is *almost* coherent — the real mechanism, measured honestly

- **CMO measures net direction, not magnitude.** `CMO = 100·(sum_up − sum_down) /
  (sum_up + sum_down)` over a trailing window is bounded by construction — it saturates
  toward ±100 whenever price moves *persistently* in one direction, however small each
  individual move is, and sits near 0 whenever up-moves and down-moves cancel, however
  *large* those moves are. "Volatile" and "trending" are not the same regime, and
  Chande's own formula only measures the second one. Tested directly on SPY
  (`strategy.regime_correlations`): VI = `|CMO|/100` correlates **+0.38** with a
  trailing trend-strength proxy but **−0.10** with trailing realized volatility — the
  "trending" half of the pitch survives, the "volatile" half does not.
- **The step-response / tracking-distance pair (as in Studies 672, 673) isolates
  mechanism from noise.** A deterministic flat-then-jump series shows VIDYA freezes
  completely before the jump (CMO ≈ 0, no signal) and then converges to *exactly* the
  same catch-up speed as an EMA of the same nominal length once the jump saturates CMO
  — VIDYA never out-runs the EMA it's raced against, it matches it. On the real tape,
  where markets spend more time in low-|CMO| chop than in the clean, saturating trend
  the step test isolates, VIDYA's *average* tracking distance to price (2.20%) is
  worse than either SMA's (1.53%) or EMA's (1.33%) — the literal "hugs price" framing
  fails on the tape even though the mechanism itself works as designed.
- **CMO itself was already tested on this desk.** [Study 185 —
  Chande Momentum](../../185-chande-momentum/) found CMO's overbought/oversold and
  zero-cross framings add no measurable timing signal over RSI/%R on daily equities
  (both HAC *t* < 0.25). VIDYA inherits CMO as its speed knob, not as a trading signal
  in its own right — a separate question this study answers on its own terms.
- **Regime-dependence of MA rules is well documented.** Brock, Lakonishok & LeBaron
  (1992), *"Simple Technical Trading Rules and the Stochastic Properties of Stock
  Returns"* (Journal of Finance), and Park & Irwin (2007), *"What Do We Know About the
  Profitability of Technical Analysis?"* (Journal of Economic Surveys), document how
  fragile MA-rule profitability is out of sample — the backdrop every "smarter MA"
  claim on this desk is measured against.

## The failure mode exposed

- **Fewer trades, not better trades.** VIDYA genuinely fires **37-40% fewer** position
  changes than SMA(14)/EMA(14) — the same direction as McGinley Dynamic (Study 672,
  −30-33%) and the opposite of the Hull MA (Study 432, +87%) and KAMA (Study 433,
  +66%). But measured head-to-head (VIDYA minus SMA, VIDYA minus EMA, HAC *t* on the
  daily spread), the edge is positive on paper on 4-5 of 5 basket tickers yet clears
  *t* = 2 on **zero of 10** such comparisons — fewer trades that are not certifiably
  better trades.
- **Timing-vs-holding is the only fair race.** All three moving-average rules — VIDYA
  included — lose to plain buy-and-hold, HAC *t* = −3.54 (SPY, net), negative on all
  five tapes, in both sample halves, and at every CMO-lookback setting from 5 to 30
  bars; a position-shuffle permutation shows the realised VIDYA calls are *worse* than
  98.85% of random re-timings of the same path. This is the alpha-vs-beta discipline
  the desk runs on every "smarter MA" claim (Studies 91, 432, 433, 434, 672, 673).
- **Parameter choice, named.** period = 14 matches the study's SMA/EMA comparators
  (same N, fair race); `cmo_period = 9` is Chande's own commonly-cited default
  lookback, and the CMO-period robustness sweep (5→30 bars) confirms the negative
  spread is not an artefact of that one canonical choice — we do not search for a
  window that "works". Sullivan, Timmermann & White (1999), *"Data-Snooping, Technical
  Trading Rule Performance, and the Bootstrap"* (Journal of Finance), is exactly the
  trap that would open if we did.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *"A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix"*
  (Econometrica) — [`strategy.hac_tstat`](../vidya/strategy.py).
- **Permutation / block-shift placebo.** Circular-shift null on the realised position
  path, in the randomization-inference tradition of Politis & Romano (1994), *"The
  Stationary Bootstrap"* (JASA) — [`strategy.permutation_pvalue`](../vidya/strategy.py).
- **Execution-lag & cost discipline.** One documented `shift` (signal on close of *t*
  earns *t+1*), costs one-way × NAV, shorts pay borrow — per
  [`METHODOLOGY.md`](../../../METHODOLOGY.md).
- **Reproducibility stamp.** `quantlab.repro.data_stamp` / `fingerprint` pin the as-of
  date and a content hash of each tape (see [`docs/results.md`](results.md)).

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`, `auto_adjust=True` → split/dividend
  adjusted, total-return bars), full history to **2026-06-30** across five liquid
  tapes (SPY, QQQ, AAPL, MSFT, XLE — the same basket as sibling studies 672/673). The
  reproducible core and the offline synthetic positive control run on
  [`data.synthetic_panel`](../vidya/data.py), never the network. Each headline is
  pinned with an as-of date and a per-tape content fingerprint (see
  [`docs/results.md`](results.md)).

## Related desk studies — the dedup map (what this study is NOT)

- **[Study 185 — Chande Momentum](../../185-chande-momentum/)**: tests CMO itself as a
  standalone overbought/oversold and zero-cross *trading signal* (found: no measurable
  edge, *t* < 0.25). This study uses CMO only as VIDYA's internal speed knob, and asks
  a different question — does *scaling an EMA's smoothing constant* by it produce a
  better moving average.
- **[Study 433 — KAMA (Kaufman Adaptive MA)](../../433-kama-adaptive/)**: the closest
  structural cousin — also an EMA whose smoothing constant adapts to a bounded [0,1]
  regime signal (Kaufman's Efficiency Ratio: net travel ÷ total path) rather than
  Chande's CMO (net direction ÷ total absolute movement over a fixed window — a
  related but distinct construction, and *not* itself a ratio of net-to-total travel
  the way ER is). KAMA's adaptation *increases* turnover (+66% vs SMA); VIDYA's
  *decreases* it (−37-40%) — opposite whipsaw effects from superficially similar
  "adapt to X, not chop" pitches.
- **[Study 672 — McGinley Dynamic](../../672-mcginley-dynamic/)**: a *different*
  self-adjusting mechanism (a quartic price/line-ratio brake, no separate oscillator
  input) that also **decreases** whipsaws (−30-33% vs SMA/EMA) — VIDYA's closest
  behavioral cousin on the desk, reached by a completely different formula, with the
  same structural verdict (no certified edge over the "dumb" MAs either claims to
  beat).
- **[Study 673 — T3 (Tillson)](../../673-t3-tillson/)**: a *fixed-coefficient*,
  six-stage nested EMA stack tuned by a constant "volume factor" — no state-dependent
  adaptation at all (T3's speed never changes with market conditions), the structural
  opposite of VIDYA's and KAMA's state-driven approach, though it lands on the same
  "cleaner crossovers, no certified edge" verdict.
- **[Study 91 — Death-Cross](../../91-death-cross/)**: a single fixed SMA(50)/SMA(200)
  crossover, tested as a bear-market dodge — not a "smarter line" claim at all.
- **[Study 432 — Hull Moving Average](../../432-hull-moving-average/)** and
  **[Study 434 — DEMA/TEMA](../../434-dema-tema/)**: lag-cancelling *extrapolation*
  filters (weighted-MA overshoot / repeated EMA differencing) that **increase**
  whipsaws — the opposite failure mode from VIDYA's and McGinley's *brake*-type
  adaptations.

None of the siblings test Chande's own volatility/trend-scaled smoothing constant, or
separate out the "volatile" claim from the "trending" claim the way this study does —
that decomposition is this study's own axis.
