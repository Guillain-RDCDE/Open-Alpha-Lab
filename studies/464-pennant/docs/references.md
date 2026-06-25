# References & literature map — Study 464 (Pennant)

## The claim under test

- **The folklore.** A **pennant** is a *continuation* pattern: a steep, near-vertical **pole**
  (a strong directional thrust) followed by a brief **converging** consolidation — a small
  symmetrical triangle with down-tilting highs and up-tilting lows on shrinking range and drying
  volume — then a **breakout in the pole's direction**. The lore: *the pennant continues the
  prior thrust*, and "the flag flies at half-mast" (price runs roughly another pole-length after
  the breakout). It is a retail/technician staple built into TradingView, MetaTrader,
  StockCharts and every chart-pattern site.
- **The source.** Robert D. **Edwards & John Magee**, *Technical Analysis of Stock Trends*
  (1948 and later editions), codified flags and pennants as short-term continuation patterns;
  John J. **Murphy**, *Technical Analysis of the Financial Markets* (1999) restates the rule for
  the modern reader. Thomas N. **Bulkowski**'s *Encyclopedia of Chart Patterns* (2005) is the
  most-cited attempt to put *measured* break-even/failure statistics on pennants and flags (and
  notably reports pennants as among the **worst**-performing classical patterns).
- **Variants & cousins.** Flags (a parallel-channel pause instead of a converging triangle),
  symmetrical/ascending/descending triangles, and wedges are affine cousins of the same
  "consolidation-then-continuation" geometry and inherit the same drift confound tested here.

## Why this is a "theory" / mechanical-proxy study

A pennant is *semi-subjective*: a discretionary chartist chooses what counts as a "steep enough"
pole, a "tight enough" triangle and a "clean" breakout. Following the desk's design for this kind,
we encode the **tightest mechanical rule a proponent would accept** and state the irreducible
subjectivity (and the missing volume leg) explicitly:

- **Objective pole.** A cumulative move over a fixed lookback exceeding a volatility-scaled
  threshold (`pole_k` × rolling σ × √pole_len) — read entirely on past bars.
- **Objective convergence.** The body's recent half-range must be a fixed fraction (`converge`)
  below its earlier half-range — a contracting triangle — with a small net move (a genuine
  pause). Measured *excluding* the breakout bar (no look-ahead).
- **Objective breakout & direction.** The close escapes the body range in the pole direction;
  the trade is long if the pole was up, short if down, entered the **next close**.
- **The honest baselines.** The only meaningful comparison on an upward-drifting index is the
  **random-entry** control (same instrument, epoch, hold, *and* long/short mix), because a
  net-long rule inherits the drift. We add a **direction-scramble placebo** that keeps the
  breakout dates but randomizes the traded direction — the direct test of "does trading *in the
  pole direction* (continuation) matter?"

Hand-drawn pennants add *hindsight* (a free parameter), which can only inflate in-sample fit; the
mechanical version is therefore the charitable **upper bound** on the method.

## Why the high one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a pole-direction
  rule that is net long ~80% of the time inherits that drift, so a one-sample *t* against **zero**
  measures the tide, not the tool. See Fama & French on the equity premium; the desk's standing
  rule is *signal-vs-baseline*, never *signal-vs-zero*.
- **Data snooping on chart patterns.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis*, *Journal of Finance*) formalize testing chart patterns against a properly matched
  null and find most carry little incremental information; Sullivan, Timmermann & White (1999,
  *Data-Snooping, Technical Trading Rule Performance, and the Bootstrap*, *JF*) and White (2000,
  *A Reality Check for Data Snooping*, *Econometrica*) show how pattern-fitted rules manufacture
  significance unless raced against a fair benchmark.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the pennant-vs-random difference.

## Method lineage (the desk's shared engine)

- **Mechanical pole + converging body + breakout.**
  [`strategy.detect_pennants`](../pennant/strategy.py) — the geometry with the no-look-ahead
  windowing baked in (pole and triangle read on bars ≤ t, entry at t+1).
- **Forward-return + HAC t + random baseline.** [`strategy.forward_returns`](../pennant/strategy.py),
  [`strategy.hac_t`](../pennant/strategy.py), [`strategy.run_experiment`](../pennant/strategy.py).
- **Direction placebo.** [`strategy.direction_placebo`](../pennant/strategy.py) — same dates,
  scrambled direction; the honest "is continuation load-bearing?" null.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../pennant/data.py) plants a real
  pole→pause→continuation pennant (knob `edge`); with `edge = 0` the detector must NOT manufacture
  significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced
  by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../450-andrews-pitchfork`](../450-andrews-pitchfork) — the sibling "price respects the
  geometry" chart-tool study with the same random-entry idiom and a geometry placebo.
- [`../../410-flag`](../../410-flag) and the broader chart-pattern / technical-indicator zoo —
  most land None × Mirage for the same reason: a pattern fitted to past price re-describes the
  trend rather than forecasting it.
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting) frame why a
  signal-vs-zero *t* is not enough; the pennant is a clean live example of a continuation label
  that adds nothing once the drift and the direction are stripped out.
