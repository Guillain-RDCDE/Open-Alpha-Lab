# References & literature map — Study 672 (McGinley Dynamic)

## The claim under test

- **The folk recipe.** John R. McGinley introduced the "Dynamic Line" in *"Trading Systems
  and Methods"*-adjacent columns for the Market Technicians Association's *Technically
  Speaking* newsletter (1990s; McGinley later summarized it on mcginleydynamic.com), sold
  as a moving average that "**automatically adjusts to market speed**" via
  `MD(t) = MD(t-1) + (P(t) - MD(t-1)) / (N * (P(t)/MD(t-1))^4)`. The pitch, repeated across
  TradingView scripts, StockCharts' technical-indicator glossary and broker education
  pages ever since: because MD "hugs" price more faithfully than a fixed SMA or EMA, a
  price-cross rule built on it turns earlier and produces **fewer whipsaws** — "a moving
  average that never has to be adjusted, no matter what market you're in or how it's
  acting" (McGinley's own copy). We steelman this as: *a McGinley Dynamic(14) price-cross
  long/flat rule fires fewer position changes than the equivalent SMA(14)/EMA(14) rule
  AND beats them (and buy-and-hold) on a net, excess-of-cash basis.*

## Why the steelman is *almost* coherent — the real mechanism, measured honestly

- **The quartic brake is a real, distinctive mechanism.** Unlike DEMA/TEMA (Study 434),
  which cancel lag by *extrapolating* price forward, McGinley's `(P/MD)^4` term is a
  **negative-feedback brake**: the further price runs from the line, the *smaller* the
  next increment. That is a genuinely different idea from every other "smarter MA" on
  this desk (91, 432, 433, 434) — and it is testable directly, without any trading rule
  at all: a deterministic step-response check (`strategy.step_response`) and a real-tape
  tracking-distance check (`strategy.tracking_distance`) both show the mechanism runs
  **backwards** from the marketing: MD tracks SPY *more loosely* (1.98% mean distance vs
  EMA's 1.33%) and reacts *slower* to a shock (20% of a step gap closed in 5 bars vs
  EMA's 58%). The brake resists exactly the fast moves a trend-follower wants to catch.
- **The one part of the pitch that survives:** McGinley Dynamic really does fire
  **30-33% fewer** position changes than SMA(14)/EMA(14) — the opposite finding from the
  desk's other "faster/adaptive MA" teardowns, where lower (perceived) lag bought *more*
  whipsaws (Hull MA, Study 432: +87%; KAMA, Study 433: +66%). The quartic brake's
  resistance to noise, not to trend, is what cuts the trade count here.
- **Regime-dependence of MA rules is well documented.** Brock, Lakonishok & LeBaron
  (1992), *"Simple Technical Trading Rules and the Stochastic Properties of Stock
  Returns"* (Journal of Finance), and Park & Irwin (2007), *"What Do We Know About the
  Profitability of Technical Analysis?"* (Journal of Economic Surveys), document how
  fragile MA-rule profitability is out of sample — the backdrop every "smarter MA" claim
  on this desk is measured against.

## The failure mode exposed

- **Fewer trades, not better trades.** Cutting whipsaws is only valuable if the
  remaining crossovers are informative. Measured head-to-head (McGinley minus SMA,
  McGinley minus EMA, HAC *t* on the daily spread), the edge is positive on paper on
  4-5 of 5 basket tickers but clears *t* = 2 on only 3 of 10 such comparisons — a coin
  that occasionally lands heads, not a certified edge.
- **Timing-vs-holding is the only fair race.** All three moving-average rules — McGinley
  Dynamic included — lose to plain buy-and-hold, HAC *t* = −3.55 (SPY, net), negative on
  all five tapes and in both sample halves; a position-shuffle permutation shows the
  realised McGinley calls are *worse* than 98.95% of random re-timings of the same path.
  This is the alpha-vs-beta discipline of the desk (Studies 91, 432, 433, 434 all reach
  the same structural verdict from different mechanisms).
- **Parameter choice, named.** N = 14 is McGinley's own stated default for daily data
  (matching the study's SMA/EMA comparators at the same N, so the race is fair); we do
  not sweep N to search for a window that "works" — Sullivan, Timmermann & White (1999),
  *"Data-Snooping, Technical Trading Rule Performance, and the Bootstrap"* (Journal of
  Finance), is exactly the trap that would open.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *"A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix"* (Econometrica) —
  [`strategy.hac_tstat`](../mcginley_dynamic/strategy.py).
- **Permutation / block-shift placebo.** Circular-shift null on the realised position
  path, in the randomization-inference tradition of Politis & Romano (1994), *"The
  Stationary Bootstrap"* (JASA) — [`strategy.permutation_pvalue`](../mcginley_dynamic/strategy.py).
- **Execution-lag & cost discipline.** One documented `shift` (signal on close of *t*
  earns *t+1*), costs one-way × NAV, shorts pay borrow — per
  [`METHODOLOGY.md`](../../../METHODOLOGY.md).
- **Reproducibility stamp.** `quantlab.repro.data_stamp` / `fingerprint` pin the as-of
  date and a content hash of each tape (see [`docs/results.md`](results.md)).

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`, `auto_adjust=True` → split/dividend
  adjusted, total-return bars), full history to **2026-06-30** across five liquid
  tapes (SPY, QQQ, AAPL, MSFT, XLE). The reproducible core and the offline synthetic
  positive control run on [`data.synthetic_panel`](../mcginley_dynamic/data.py), never
  the network. Each headline is pinned with an as-of date and a per-tape content
  fingerprint (see [`docs/results.md`](results.md)).

## Related desk studies — the dedup map (what this study is NOT)

- **[Study 91 — Death-Cross](../../91-death-cross/)**: a single fixed SMA(50)/SMA(200)
  crossover, tested as a bear-market dodge — not a "smarter line" claim at all. Different
  question (risk reduction vs a coin), same honest arbiter.
- **[Study 432 — Hull Moving Average](../../432-hull-moving-average/)**: a *different*
  low-lag mechanism (weighted-MA extrapolation) that **increases** whipsaws (+87% vs
  SMA) — the opposite failure mode from McGinley's brake, which **decreases** them.
- **[Study 433 — KAMA (Kaufman Adaptive MA)](../../433-kama-adaptive/)**: adapts its
  smoothing constant to an *efficiency ratio* (trend-vs-chop), also **increasing**
  turnover (+66%) rather than cutting it. McGinley's quartic brake is a structurally
  different adaptation rule with the opposite whipsaw effect.
- **[Study 434 — DEMA & TEMA](../../434-dema-tema/)**: lag-cancelling extrapolation
  filters — *less* lag, *more* whipsaws, *worse* rule. McGinley Dynamic is the mirror
  case: *more* lag (it reacts slower, not faster, per the step-response test), *fewer*
  whipsaws — and still no certified edge. Together the four studies show whipsaw count
  and lag are not the lever that matters; none of the four "smarter" lines beats
  buy-and-hold or a coin.
- **[Study 437 — Triple MA Crossover](../../437-triple-ma-crossover/)**: a *rule-shape*
  variant (three lines, a confirmation cascade) rather than a *line-shape* variant — a
  different lever entirely from what any single indicator's curve looks like.

None of the siblings test John McGinley's specific self-adjusting formula or its
quartic-brake mechanism — that is this study's own axis.
