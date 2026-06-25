# References & literature map — Study 468 (Gartley / AB=CD Harmonic)

## The claim under test

- **The folklore.** A five-point zig-zag **X-A-B-C-D** of swing pivots is a "harmonic pattern"
  when its leg *retracement ratios* land on the Fibonacci grid: for a bullish **Gartley**, B
  retraces ≈ **0.618** of XA, C retraces 0.382–0.886 of AB, and D retraces ≈ **0.786** of XA (the
  "AB=CD" symmetry has CD ≈ AB). When the grid is satisfied, point **D is a high-probability
  reversal** — a buy in an up-context. This is the retail/technician staple sold by harmonic-
  trading courses and built into TradingView (auto-harmonic scanners), MetaTrader and Thinkorswim.
- **The source.** **H. M. Gartley**, *Profits in the Stock Market* (1935), drew the original
  five-point figure (his "Gartley 222"). The *Fibonacci-ratio* codification is later: **Larry
  Pesavento**, *Fibonacci Ratios with Pattern Recognition* (1997), and **Scott M. Carney**, *The
  Harmonic Trader* (1998) / *Harmonic Trading, Vols. 1–2* (2010), who named the modern zoo
  (Gartley, Bat, Butterfly, Crab) and fixed the exact ratios. Bryce Gilmore's *Geometry of
  Markets* is part of the same lineage.
- **Variants.** Bat (D = 0.886·XA), Butterfly (D = 1.272·XA), Crab (D = 1.618·XA), Cypher and
  Shark are **the same XABCD geometry with different Fibonacci targets** and inherit the same
  ratio-as-free-parameter problem tested here.

## Why this is a "theory" / mechanical-proxy study

Harmonic-pattern reading is *semi-subjective*: a discretionary trader chooses which swings to
label X-A-B-C-D and how loose a tolerance to accept. Following the desk's design for this kind, we
encode the **tightest mechanical rule a proponent would accept** and state the irreducible
subjectivity explicitly:

- **Objective pivots.** Confirmed **fractals** (Bill Williams' definition: a local extremum with
  *k* strictly-lower/higher bars on each side), only usable *k* bars later — a documented
  confirmation lag, no look-ahead.
- **Objective pattern.** At each confirmed swing-low we scan recent pivots for a correctly-ordered,
  alternating XABCD whose ratios satisfy the Gartley grid within a tolerance — no hand-labelling.
- **The honest baseline.** The only meaningful comparison on an upward-drifting index is the
  **random-entry** control (same instrument, epoch and hold), because *any* dip-buy inherits the
  drift. We add a **ratio-grid placebo** that swaps the Fibonacci targets for random ratios while
  keeping the zig-zag machinery — the direct test of "do the *Fibonacci* ratios matter?"

A looser, hand-labelled harmonic scan adds *hindsight* (free parameters), which can only inflate
in-sample fit; the mechanical version is therefore the charitable **upper bound** on the method.

## Why the high one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample *t*
  of a long-only entry rule against **zero** measures that drift, not the rule. See Fama & French
  on the equity premium; the desk's standing rule is *signal-vs-baseline*, never *signal-vs-zero*.
- **Data snooping on chart tools.** Lo, Mamaysky & Wang (2000, *Foundations of Technical Analysis*,
  *Journal of Finance*) formalize testing chart patterns against a properly matched null and find
  most "patterns" add little once the benchmark is fair; Sullivan, Timmermann & White (1999,
  *Data-Snooping, Technical Trading Rule Performance, and the Bootstrap*, *Journal of Finance*) and
  White (2000, *A Reality Check for Data Snooping*, *Econometrica*) show how rules selected from a
  large family (the harmonic zoo is exactly such a family) manufacture significance unless raced
  against the whole search space. The **ratio-grid placebo** is our local Reality Check: it asks
  whether the *Fibonacci* ratios beat the family of *all* plausible ratios.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the D-vs-random difference.

## Method lineage (the desk's shared engine)

- **Confirmed-fractal pivots + windowed XABCD scan.**
  [`strategy.find_pivots`](../gartley_harmonic/strategy.py),
  [`strategy.detect_completions`](../gartley_harmonic/strategy.py) — the mechanical geometry with
  the confirmation lag baked in.
- **Forward-return + HAC t + random baseline.**
  [`strategy.forward_returns`](../gartley_harmonic/strategy.py),
  [`strategy.hac_t`](../gartley_harmonic/strategy.py),
  [`strategy.run_experiment`](../gartley_harmonic/strategy.py).
- **Fibonacci-ratio placebo.** [`strategy.ratio_grid_placebo`](../gartley_harmonic/strategy.py) —
  swap the Gartley grid for random ratios, keep the zig-zag machinery and marginals.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../gartley_harmonic/data.py)
  splices real Gartley zig-zags with a planted D-point bounce (knob `edge`); with `edge = 0` the
  detector must NOT manufacture significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../450-andrews-pitchfork`](../450-andrews-pitchfork) — the same three/five-point geometric
  channel folklore, tested with the random-entry baseline + geometry-scramble placebo idiom.
- [`../410-cup-and-handle`](../410-cup-and-handle), [`../413-bull-flag`](../413-bull-flag) and the
  broader chart-figure zoo — the same "a named shape forecasts a turn" claim; the ratio-grid
  placebo here is the cleanest refutation of the Fibonacci magic-number premise.
- The **research-method demos** (data-mining-roulette, multiple-testing, curve-fitting) frame why a
  signal-vs-zero *t* is not enough; the harmonic D-point is a clean live example of a long-horizon
  dip-buy wearing a Fibonacci costume.
