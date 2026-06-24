# References & literature map — Study 413 (Bull Flag)

## The claim under test

- **The folklore.** "After a sharp rally — the **flagpole** — a stock pauses in a brief, shallow
  consolidation that drifts gently down or sideways in a tight channel — the **flag**. When it
  closes back above the flag's high (the **breakout**), the prior uptrend *resumes* for a second leg,
  often measured as 'another flagpole's worth.' So buy the breakout and ride it." It is one of the
  most-taught bullish *continuation* figures in retail technical analysis.
- **The classic sources.** Robert D. Edwards & John Magee, *Technical Analysis of Stock Trends*
  (1948 and later editions) — the canonical taxonomy of flags, pennants and other continuation
  figures. Thomas N. Bulkowski, *Encyclopedia of Chart Patterns* (2nd ed., 2005) catalogues the
  "high and tight flag" and ordinary flags with hand-counted performance and failure-rate
  statistics, and is the most-cited modern reference for chartists making the bull-flag claim.
- **The skeptical literature.** Andrew Lo, Harry Mamaysky & Jiang Wang, *Foundations of Technical
  Analysis* (2000, Journal of Finance) built the first rigorous kernel-smoothing detector for chart
  figures (head-and-shoulders, flags, etc.) and found *some* patterns carry marginal information but
  far less than the folklore claims. Park & Irwin (2007, *What do we know about the profitability of
  technical analysis?*, Journal of Economic Surveys) survey decades of evidence: early studies often
  found profits that vanished under transaction costs, data-snooping corrections, and out-of-sample
  testing. Our negative result is squarely in this tradition.

## Why a mechanical detector — and its honest limits

- **Chart figures are partly subjective.** A human chartist draws a flag with judgement about what
  counts as "the pole," "tight enough," and "a clean break." We test the closest *mechanical*
  definition we can write down — a steep ``pole_w``-bar run-up clearing ``min_pole``, a
  ``flag_min..flag_max``-bar consolidation with bounded retrace / range / drift, and a confirmed
  close above the flag high — and we say so loudly. A different rule set finds a different
  (overlapping) event set; the *qualitative* verdict (direction is uninformative) is what travels.
- **The detector mirrors the folk recipe.** [`strategy.detect_flags`](../bull_flag/strategy.py)
  encodes pole → flag → breakout; the ``side="down"`` branch runs the *identical* shape but requires
  a break below the flag's low — the symmetry myth-check.

## The inference — why the placebo, not the *t*, is the arbiter

- **Excess over the name's own base rate.** A naive forward-return *t* on a breakout that *selects
  post-run-up dates* is confounded by the names' own up-drift. We subtract each name's unconditional
  mean forward return ([`strategy.base_rate`](../bull_flag/strategy.py)) so the test is "does the
  figure beat buy-and-hold *for that name*."
- **Same-tape label-shuffle placebo.** [`strategy.run_experiment`](../bull_flag/strategy.py) draws
  random entry dates on the *same* tape (same count per name) and asks how often a random set beats
  the observed mean — the honest control for residual drift/momentum the *t* misses. This is the
  Lo-Mamaysky-Wang spirit and the desk's standard for selection-on-a-rule (Harvey, Liu & Zhu 2016,
  *…and the Cross-Section of Expected Returns*, RFS; White 2000 Reality Check logic).
- **HAC / one-sample *t*.** [`strategy.hac_t`](../bull_flag/strategy.py) — Newey-West (1987)
  autocorrelation-robust standard errors for clustered breakouts. Here it agrees with the naive *t*
  (both ≈ −1.1) because the up-break excess is simply negative.

## Method lineage (the desk's shared engine)

- **Deterministic synthetic control.** [`data.synthetic_panel`](../bull_flag/data.py) plants clean
  bull flags and a *known* post-breakout drift; with the edge set to zero the inference must NOT
  manufacture significance (it doesn't — placebo *p* = 0.66), and a planted +20% drift must light up
  (*t* = 16.4) — the positive control that proves the engine, not the market, is sound.
- **Sibling chart-figure studies on this desk.** This study is the direct twin of the
  ascending-triangle and triangle teardowns — same detector idiom, same excess-over-base-rate +
  same-tape placebo + down-break symmetry arbiters.

## Data sources used here

- **yfinance** daily *auto-adjusted* (split + dividend) OHLC for SPY + 29 long-listed US large-caps,
  2005-01-03 → 2026-05-29 (as-of 2026-05-31), cached under `_cache/bullflag_{open,high,low,close}.parquet`.
  All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../410-cup-and-handle`](../410-cup-and-handle) — the cup-and-handle continuation figure.
- [`../411-ascending-triangle`](../411-ascending-triangle) — the ascending-triangle continuation
  figure (the closest sibling; WEAK × MIRAGE, breakout direction also busted).
- [`../412-symmetrical-triangle`](../412-symmetrical-triangle) — the symmetrical-triangle figure.
- The **research-method demos** (data-mining-roulette, multiple-testing, look-ahead) frame why a
  naive *t* alone is not enough — the bull flag is the counter-example that fails even *before* the
  placebo, because the raw edge is already negative.
