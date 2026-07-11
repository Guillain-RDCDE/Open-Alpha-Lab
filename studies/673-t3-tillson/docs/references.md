# References & literature map — Study 673 (T3, Tillson)

## The claim under test

- **The recipe.** Tim Tillson, *"Better Moving Averages"*, **Technical Analysis of Stocks &
  Commodities**, January 1998. Tillson defines a "generalized DEMA" (GD) — a tunable blend
  of an EMA and a double-EMA controlled by a **volume factor** v (his own term; nothing to
  do with traded share volume) — and then nests GD **three times** to build "T3":
  `GD(x, v) = (1+v)*EMA(x) − v*EMA(EMA(x))`, `T3 = GD(GD(GD(x, v), v), v)`. Expanded, T3 is
  a fixed linear combination of the 3rd through 6th EMA of price (six ordinary EMA passes,
  no recursion in price itself — see [`t3_tillson/strategy.py`](../t3_tillson/strategy.py)).
- **The pitch.** Tillson's own abstract, echoed verbatim on TradingView scripts, MT4/5
  indicator libraries and broker education pages ever since: T3 "virtually eliminates lag
  while smoothing the data," so a crossover or slope rule built on it should turn earlier
  **and** whipsaw less than a plain SMA/EMA of the same nominal length. We steelman this as:
  *a T3 price-cross (or T3-slope) long/flat timing rule on daily equity bars beats
  buy-and-hold on a net, excess-of-cash Sharpe basis, beats the equivalent SMA/EMA rule,
  and does so with fewer whipsaws.*

## Why the steelman is coherent on paper — the real thing it leans on

- **T3 is genuinely smoother, not just marketing.** Six stacked EMA passes really do damp
  high-frequency noise more than a single SMA/EMA of the same N — that shows up directly as
  **fewer position switches per year** on every basket ticker in this study (12–17% fewer
  than SMA/EMA). Tillson's own comparison (T3 vs DEMA/TEMA of the *same* N) is legitimate:
  T3 removes the overshoot those faster filters carry.
- **Moving-average trend rules can work — on trending series.** A price-vs-MA rule is a
  trend-following filter, and trend-following is a documented premium in some markets and
  horizons (Moskowitz, Ooi & Pedersen 2012, *"Time Series Momentum"*, Journal of Financial
  Economics; Hurst, Ooi & Pedersen 2017, *"A Century of Evidence on Trend-Following
  Investing"*, AQR). The T3 folklore borrows this legitimacy the same way every "smarter MA"
  claim on this desk does.
- **The lag-vs-noise trade-off is the classic result the pitch glosses over** (Ehlers 2001,
  *"Rocket Science for Traders"*): you cannot reduce lag and reduce noise sensitivity for
  free. T3's actual trade is the *opposite* direction from what its abstract advertises when
  measured against a plain SMA/EMA of the same N — see the failure mode below.

## The failure mode exposed

- **"Virtually eliminates lag" does not survive a step-response test.** At the same nominal
  N, T3(14, v=0.7) tracks price *less* tightly than SMA(14) or EMA(14) (mean tracking
  distance 1.88% vs 1.53%/1.33%) and is the *slowest* of the three to catch up after a
  deterministic +20% step. The mechanism: T3 nests its lag-correction three times, so by the
  third nesting the correction is being applied to an already doubly-smoothed input, not raw
  price — six stages of smoothing dominate the algebraic extrapolation at a shared N. This
  is a structural property, reproduced directly from Tillson's own published formula, not an
  artefact of this implementation.
- **"Fewer whipsaws" is real but doesn't pay.** T3 fires fewer position switches than SMA/EMA
  on all five basket tickers — the literal claim holds — but the head-to-head active-spread
  HAC *t* against SMA (+0.97) and EMA (+0.42) never clears the desk's *t* ≥ 2 bar. Smoother
  timing is not the same as *better* timing.
- **Timing-vs-holding is the only fair race.** A long-biased rule in a multi-decade bull
  tape makes money; that is exposure, not skill. Measuring the *active spread* (strategy −
  buy&hold) and the position-shuffle permutation isolates the timing, and the timing is
  significantly negative (HAC *t* = −4.03, permutation *p* = 0.927 — worse than 92.7% of
  random re-timings of the same position path). This is the alpha-vs-beta discipline of the
  desk.
- **Out-of-sample / data-snooping.** N=14 and v=0.7 are Tillson's own conventional defaults
  (and the desk's "same nominal N" convention shared with siblings 432/672); sweeping v to
  find the one value that "works" would be exactly the data-snooping that Sullivan,
  Timmermann & White (1999), *"Data-Snooping, Technical Trading Rule Performance, and the
  Bootstrap"* (Journal of Finance), and Bailey, Borwein, López de Prado & Zhu (2014),
  *"Pseudo-Mathematics and Financial Charlatanism"* (Notices of the AMS), warn against —
  every v from 0.1 to 0.9 tested here is negative and significant, so there is no v to
  snoop toward. Brock, Lakonishok & LeBaron (1992), *"Simple Technical Trading Rules and the
  Stochastic Properties of Stock Returns"* (Journal of Finance), and Park & Irwin (2007),
  *"What Do We Know About the Profitability of Technical Analysis?"* (Journal of Economic
  Surveys), document how fragile MA-rule profitability is out of sample generally.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *"A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix"* (Econometrica) —
  implemented in [`strategy.hac_tstat`](../t3_tillson/strategy.py).
- **Permutation / placebo testing.** The circular-shift placebo on the realised position
  path follows the randomisation-inference tradition (Politis & Romano 1994, *"The
  Stationary Bootstrap"*, JASA) — [`strategy.permutation_pvalue`](../t3_tillson/strategy.py).
- **Execution-lag & cost discipline.** One documented `shift` (signal on close of *t* earns
  *t+1*), costs one-way × NAV, shorts pay borrow — per [`METHODOLOGY.md`](../../../METHODOLOGY.md).

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), full history through **2026-06-30**,
  across five liquid total-return tapes (SPY, QQQ, AAPL, MSFT, XLE). The offline
  reproducible core and the notebooks run on the deterministic
  [`data.synthetic_panel`](../t3_tillson/data.py) generator when no cache is present, never
  the network. Each headline is pinned with an as-of date and a per-tape content
  fingerprint (see [`docs/results.md`](results.md)).

## Related desk studies (the dedup map — what this study is NOT)

- **[Study 672 — McGinley Dynamic](../../672-mcginley-dynamic/)**: a *recursive*,
  price-adaptive MA (its own update speed changes with the price/line ratio) — T3 is a
  *fixed* linear recombination of six ordinary EMAs; the adaptivity claims are structurally
  different mechanisms, tested with the same honest race.
- **[Study 432 — Hull Moving Average](../../432-hull-moving-average/)**: the opposite
  failure mode. HMA genuinely lowers lag and consequently fires **more** whipsaws (32.5 vs
  17.4/yr) than its SMA comparator — noise-chasing. T3 genuinely raises smoothness and fires
  **fewer** whipsaws (31.9 vs 36.5/yr) — but is *slower*, not faster, to react at the same N.
  Contrasting the two failure modes is the point of running both studies.
- **[Study 483 — ZLEMA](../../483-zlema/)**: a *two-term* lag-cancellation EMA (single
  correction term, not T3's six-stage nested correction) — the "minimal" end of the same
  lag-reduction family; T3 is deliberately the "maximal smoothing" end.
- **[Study 674 — VIDYA](../../674-vidya/)**: a *volatility-adaptive* EMA (its span changes
  with realised Chande Momentum, not a fixed volume factor) — adaptivity driven by market
  state, not by a hand-tuned constant like T3's v.
- **[Study 433 — KAMA (Adaptive)](../../433-kama-adaptive/)**: Kaufman's efficiency-ratio
  adaptive EMA — another *state-driven* adaptive-speed MA, the closest philosophical cousin
  to VIDYA rather than to T3's fixed-coefficient smoothing stack.

None of the siblings test T3's specific mechanism (six nested EMA stages recombined by a
fixed volume factor) or its specific, counter-intuitive failure mode (smoother *and*
slower, at the same nominal N) — this study is T3's own axis.
