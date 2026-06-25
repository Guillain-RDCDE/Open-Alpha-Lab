# References & literature map — Study 500 (Polarity-Flip)

## The claim under test

- **The folklore.** The *polarity principle* (a.k.a. **role reversal**, "support and resistance
  switch roles"): once price decisively **breaks above** a prior swing-high resistance level,
  that old ceiling is supposed to *flip* and act as a **floor** — so the first pullback down to a
  freshly broken resistance is a high-probability **buy** (it should hold as support and bounce).
  This is a staple of every classic charting text and retail trading course, drawn on
  TradingView, MetaTrader, Thinkorswim and StockCharts.
- **The source.** The idea is old technical-analysis lore. **Edwards & Magee**, *Technical
  Analysis of Stock Trends* (1948) — the canonical text — states the principle explicitly: a
  broken resistance "becomes support" and vice-versa. **John J. Murphy**, *Technical Analysis of
  the Financial Markets* (1999), and **Thomas Bulkowski**, *Encyclopedia of Chart Patterns*
  (2000), restate and tabulate it. **Charles Dow**'s early commentary on prior highs/lows is the
  upstream lineage. The modern popular write-ups (Investopedia "Support and Resistance Reversal",
  Babypips "Role Reversal") repeat the rule verbatim.
- **Variants.** Round-number levels, prior-day/prior-week highs, gap edges and pivot-point levels
  are all treated as the same kind of "memory" level; the role-reversal claim is applied to each.
  They are affine relatives of the same idea (a horizontal price level acquires a magnetic /
  reflective property) and inherit the same drift confound tested here.

## Why this is a "theory" / mechanical-proxy study

The polarity principle is *semi-subjective*: a discretionary trader chooses which swing high is
"the" resistance and what counts as a "decisive" break or a "clean" retest. Following the desk's
design for this kind, we encode the **tightest mechanical rule a proponent would accept** and
state the irreducible subjectivity explicitly:

- **Objective levels.** Confirmed **swing-high fractals** (a local maximum with *k* strictly-lower
  bars on each side), only usable *k* bars later — a documented confirmation lag, no look-ahead.
- **Objective break + retest.** A level is "broken" when the close exceeds it by a fixed buffer
  (+0.5%); the entry is the **first** close back inside a ±1% band around the broken level. No
  hand-picking which level or which retest.
- **The honest baseline.** The only meaningful comparison on an upward-drifting index is the
  **random-entry** control (same instrument, epoch and hold), because *any* dip-buy inherits the
  drift. We add a **scrambled-level placebo** that permutes which price sits at which pivot,
  destroying "this specific broken level" while keeping the price marginal — the direct test of
  "does the level matter?"

Hand-tuned band widths and break buffers add *hindsight* (free parameters), which can only inflate
in-sample fit; the mechanical version here is the charitable upper bound on the method.

## Why the high one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample *t*
  of a long-only entry rule against **zero** measures that drift, not the rule. See Fama & French
  on the equity premium; the desk's standing rule is *signal-vs-baseline*, never *signal-vs-zero*.
  Here the one-sample *t* is large at every horizon, but the random baseline soaks up most of it —
  only the 5-day residual clears *t* = 2.
- **Data snooping on chart tools.** Lo, Mamaysky & Wang (2000, *Foundations of Technical Analysis:
  Computational Algorithms, Statistical Inference, and Empirical Implementation*, Journal of
  Finance) formalize testing chart patterns against a properly matched null and find horizon-
  dependent, fragile information at best. Sullivan, Timmermann & White (1999, *Data-Snooping,
  Technical Trading Rule Performance, and the Bootstrap*, JF) and White (2000, *A Reality Check for
  Data Snooping*, Econometrica) show how level-fitted rules manufacture significance unless raced
  against a fair benchmark — exactly why a single-horizon hit (5-day, *p* = 0.040) across a 4-
  horizon search is treated cautiously here (Weak, not Real).
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the retest-vs-random difference.

## Method lineage (the desk's shared engine)

- **Confirmed-fractal levels + break/retest state machine.**
  [`strategy.find_swing_highs`](../polarity_flip/strategy.py),
  [`strategy.polarity_entries`](../polarity_flip/strategy.py) — the mechanical geometry with the
  confirmation lag baked in.
- **Forward-return + HAC t + random baseline.**
  [`strategy.forward_returns`](../polarity_flip/strategy.py),
  [`strategy.hac_t`](../polarity_flip/strategy.py),
  [`strategy.run_experiment`](../polarity_flip/strategy.py).
- **Level placebo.** [`strategy.scrambled_level_placebo`](../polarity_flip/strategy.py) — permute
  level prices, keep positions and marginals.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../polarity_flip/data.py) plants a
  real role-reversal bounce (knob `edge`); with `edge = 0` the detector must NOT manufacture
  significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../450-andrews-pitchfork`](../450-andrews-pitchfork) — the sibling "price respects this line"
  chart-geometry study, tested with the same random-entry + geometry-placebo idiom (landed
  None × Mirage). Polarity-flip is the rarer case where a short-horizon residual actually clears
  the bar.
- [`../104-bollinger-reversion`](../104-bollinger-reversion) — the same "price reverts at a level"
  folklore tested against the random-entry baseline.
- The **research-method demos** (data-mining-roulette, multiple-testing, look-ahead) frame why a
  single significant horizon out of four is treated as *Weak* rather than *Real*: one hit in a
  4-test family is exactly the multiple-comparisons trap they dramatize.
