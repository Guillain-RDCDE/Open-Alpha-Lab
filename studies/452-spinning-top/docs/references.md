# References & literature map — Study 452 (Spinning-Top)

## The claim under test

- **The folklore.** A *spinning top* is a candle with a **small real body** sitting between two
  **long, comparable wicks** (shadows). The body's smallness says the session opened and closed
  near the same level; the two long wicks say price probed both up and down but settled in the
  middle. The standard reading, in every candlestick primer, is **indecision** — buyers and
  sellers fought to a draw — which *resolves* into a directional move or a reversal, so the bar
  is treated as an early warning that "something is about to happen." It is built into every
  charting suite's candlestick scanner.
- **The source.** Japanese candlestick charting is traditionally traced to the rice-trading lore
  of **Munehisa Homma** (18th c.); it was introduced to Western markets and systematised by
  **Steve Nison**, *Japanese Candlestick Charting Techniques* (1991), who catalogues the spinning
  top (and the related doji) as classic "indecision" candles. Gregory Morris,
  *Candlestick Charting Explained* (1992/2006), and Thomas Bulkowski's *Encyclopedia of
  Candlestick Charts* (2008) give the modern statistical write-ups; Investopedia and StockCharts'
  ChartSchool restate the rule.
- **Variants / cousins.** The **doji** (near-zero body) is the limiting case; the **high-wave
  candle** is a spinning top with even longer shadows. All are **the same small-body / two-wick
  geometry** and inherit the same drift confound and the same "is the shape load-bearing?"
  question tested here.

## Why this is a "theory" / mechanical-proxy study

The spinning top is *semi-subjective*: traders eyeball "small body" and "long, balanced wicks"
and often demand surrounding context (a prior trend, a confirmation bar). Following the desk's
design for this kind, we encode the **tightest mechanical rule a proponent would accept** and
state the irreducible thresholds explicitly:

- **Objective body/wick test.** body/range `< 0.25`, both wicks `≥ 0.25 ×` range, wick balance
  (min/max) `≥ 0.5` — the canonical 25% body cutoff, no eyeballing.
- **No look-ahead.** All four prices are known on the bar's close; the long is entered at the
  **next** close (one documented lag).
- **The honest baseline.** The only meaningful comparison on an upward-drifting index is the
  **random-entry** control (same instrument, epoch and hold), because *any* long entry inherits
  the drift. We add a **wick-scramble placebo** that destroys the spinning-top shape (re-pairs
  bodies with permuted wick lengths) while keeping the price path and the wick marginal — the
  direct test of "does the *shape* matter?"
- **Seed-robust Welch.** Because the random baseline is a *draw* of dates, a single seed can flatter
  or flatten the rule; we average the touch-vs-random Welch *t* over 20 baseline seeds and report
  the spread, so a lucky comparison set cannot masquerade as an edge.

## Why the high one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample
  *t* of a long-only entry rule against **zero** measures that drift, not the rule. See Fama &
  French on the equity premium; the desk's standing rule is *signal-vs-baseline*, never
  *signal-vs-zero*. (Here the one-sample *t* hits +8.35 at 20d yet the rule barely beats random.)
- **Data snooping on chart patterns.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis: Computational Algorithms, Statistical Inference, and Empirical Implementation*, JF)
  formalise testing chart patterns against a properly matched null and find most patterns add
  little once the unconditional return is controlled. Sullivan, Timmermann & White (1999,
  *Data-Snooping, Technical Trading Rule Performance, and the Bootstrap*, JF) and White (2000,
  *A Reality Check for Data Snooping*, Econometrica) show how a tested-many-times rule
  manufactures significance unless raced against a fair benchmark — exactly the spinning top's
  trap.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the top-vs-random difference.

## Method lineage (the desk's shared engine)

- **Mechanical candle classification.** [`strategy.candle_parts`](../spinning_top/strategy.py),
  [`strategy.is_spinning_top`](../spinning_top/strategy.py) — body/range/wick geometry, no
  discretion.
- **Forward-return + HAC t + random baseline.** [`strategy.forward_returns`](../spinning_top/strategy.py),
  [`strategy.hac_t`](../spinning_top/strategy.py), [`strategy.run_experiment`](../spinning_top/strategy.py).
- **Geometry placebo.** [`strategy.wick_scramble_placebo`](../spinning_top/strategy.py) — permute
  wick lengths across bars, keep bodies/prices and the wick marginal.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../spinning_top/data.py) plants a
  real post-spinning-top resolution (knob `edge`); with `edge = 0` the detector must NOT
  manufacture significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) OHLC for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced
  by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../../450-andrews-pitchfork`](../../450-andrews-pitchfork) — the same drift-matched random
  baseline + geometry-scramble placebo idiom on a chart tool; lands None × Mirage for the same
  reason.
- The broader **candlestick / pattern zoo** (doji, hammer, engulfing, stars, soldiers) — most
  land None × Mirage: a single-bar shape re-describes recent volatility, not future direction.
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting, multiple-testing)
  frame why a signal-vs-zero *t* — or a single lucky baseline draw — is not evidence; the
  spinning top is a clean live example of beta plus a fragile, seed-dependent blip masquerading
  as a forecast.
