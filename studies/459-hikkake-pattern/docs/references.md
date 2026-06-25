# References & literature map — Study 459 (Hikkake pattern)

## The claim under test

- **The folklore.** The **hikkake** (Japanese 仕掛け, "trap" / "to ensnare") is a two-stage
  false-breakout pattern: an **inside bar** (a bar whose range sits entirely inside the prior
  bar's), then a **false breakout** beyond that inside range one way, then a **snap-back**
  close through the range in the opposite direction. The lore says the trap forecasts a move in
  the **reversal** direction — long after a failed downside break, short after a failed upside
  break. It is a staple of price-action / candlestick trading, taught on every chart-pattern
  site and built into many scanners.
- **The source.** **Daniel L. Chesler, CMT** introduced and named the hikkake in his article
  *"Quantifying Market Reversal Patterns"* (a.k.a. the hikkake write-ups, c. 2003–2004; see
  *Active Trader* magazine and chesler.us). It builds on the **inside-bar** concept and the
  **false-breakout / bull- and bear-trap** idea long present in technical analysis (e.g.
  Joe Ross's "Ross hook", and the inside-bar literature in Steve Nison's candlestick canon).
  Modern restatements: Investopedia's "Hikkake Pattern", StockCharts ChartSchool, and most
  price-action courses.
- **Variants.** The "modified hikkake" extends the confirmation window, and some traders add a
  trend filter or require a momentum confirmation. All are **parameter tweaks of the same
  inside-bar + false-break + snap-back geometry** and inherit the same confounds tested here.

## Why this is a "theory" / mechanical-proxy study

The hikkake is *semi-subjective*: traders argue over the confirmation window and whether to add
a trend filter. Following the desk's design for this kind, we encode the **tightest mechanical
rule a proponent would accept** and state the irreducible choices explicitly:

- **Objective inside bar.** ``high_i < high_{i-1}`` and ``low_i > low_{i-1}`` — no eyeballing.
- **Objective trap.** Within a fixed 3-bar window, a close beyond the inside range followed by
  a snap-back close through it; the snap-back close *completes* the pattern (read on its own
  close), and entry is the **next close** — a documented one-bar lag, no look-ahead.
- **The honest baseline.** Because the signal is **mixed long/short**, the only meaningful
  control matches the same **long/short mix** as well as instrument/epoch/hold (so it carries
  the same net exposure to drift). We add a **scrambled-direction placebo** that keeps every
  trap date but randomly flips the trade direction — the direct test of "does the trap's
  *direction* matter?".

## Why the one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample
  *t* of a directional rule against **zero** measures net exposure to that drift, not a
  forecast. A short-heavy rule will show a *negative* one-sample *t* simply by fading the tape.
  The desk's standing rule is *signal-vs-baseline*, never *signal-vs-zero*. See Fama & French
  on the equity premium.
- **Data snooping on chart patterns.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis: Computational Algorithms, Statistical Inference, and Empirical Implementation*,
  *Journal of Finance*) formalize testing chart patterns against a properly matched null;
  Sullivan, Timmermann & White (1999, *Data-Snooping, Technical Trading Rule Performance, and
  the Bootstrap*, *Journal of Finance*) and White (2000, *A Reality Check for Data Snooping*,
  *Econometrica*) show how price-fitted rules manufacture apparent significance unless raced
  against a fair benchmark.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the hikkake-vs-random difference.

## Method lineage (the desk's shared engine)

- **Inside-bar + false-break detection.** [`strategy.hikkake_signals`](../hikkake_pattern/strategy.py) —
  the mechanical trap geometry with the snap-back confirmation and entry lag baked in.
- **Direction-signed forward-return + HAC t + exposure-matched random baseline.**
  [`strategy.forward_returns`](../hikkake_pattern/strategy.py),
  [`strategy.hac_t`](../hikkake_pattern/strategy.py),
  [`strategy.random_entries`](../hikkake_pattern/strategy.py),
  [`strategy.run_experiment`](../hikkake_pattern/strategy.py).
- **Direction placebo.** [`strategy.scrambled_direction_placebo`](../hikkake_pattern/strategy.py) —
  keep the trap dates, flip the directions.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../hikkake_pattern/data.py)
  plants a real trap-reversal (knob `edge`); with `edge = 0` the detector must NOT manufacture
  significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced
  by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../450-andrews-pitchfork`](../450-andrews-pitchfork) — same "the geometry forecasts price"
  folklore, same random-entry + geometry-placebo idiom; also None × Mirage × Busted.
- The candlestick/figure zoo (head-and-shoulders, double-top, the morning/evening stars, the
  three soldiers, NR7) — most land None × Mirage because a pattern fitted to past bars merely
  re-describes the trend.
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting, multiple-
  testing) frame why a signal-vs-zero *t* is not enough; the hikkake is a clean live example of
  a named trap that turns out to be a coin flip once the direction is scrambled.
