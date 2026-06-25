# References & literature map — Study 477 (Choppiness Index)

## The claim under test

- **The folklore.** The **Choppiness Index** (CI) classifies the last *N* bars as *trending*
  (low CI) or *choppy/range-bound* (high CI). The retail teaching is that a **low** CI marks the
  onset of a directional "trending" regime that **precedes tradable momentum** — so you switch
  on a trend-following stance (here: go long, riding the prevailing up-trend) — while a **high**
  CI warns of whipsaw and chop. CI is built into TradingView, MetaTrader, Thinkorswim and most
  charting suites, usually with Fibonacci-flavoured 38.2 / 61.8 trend/chop bands.
- **The source.** The Choppiness Index was developed by Australian commodity trader **E. W.
  (Bill) Dreiss** in the 1990s as part of his fractal/"Choppiness" market-geometry work. The
  formula is `CI = 100·log₁₀(Σ ATR / (max(high) − min(low))) / log₁₀(N)`: it compares the summed
  per-bar true range (path length) to the high-low span of the window, normalised to ~0-100. A
  straight, directional move has Σ TR ≈ span ⇒ low CI; a back-and-forth thrash has Σ TR ≫ span
  ⇒ high CI. It is **non-directional by construction** (sign-blind), so any tradable rule must
  graft a directional bias on top.
- **Variants.** Threshold choices (38.2/61.8 vs 30/70), window length (14 the common default),
  and pairing CI with ADX/Bollinger-bandwidth as a "regime filter" are all affine/parametric
  tweaks of the same volatility-geometry gauge and inherit the same drift confound tested here.

## Why this is a "mechanical-proxy" study

The Choppiness Index is an objective number, but the *trading rule* layered on it ("low CI ⇒
the trend continues ⇒ go long") is the discretionary part. Following the desk's design, we
encode the **tightest mechanical rule a proponent would accept** and state the irreducible
choices explicitly:

- **Objective CI.** Trailing window only (`min_periods = N`), so the reading at bar *t* uses no
  future bars; a signal fires on the **onset** of a low-CI episode (first bar CI < 38.2), not
  every day it stays low.
- **Objective entry.** Long at the **next close** (one documented lag); hold 5/10/20/60 days.
- **The honest baseline.** The only meaningful comparison on an upward-drifting index is the
  **random-entry** control (same instrument, epoch and hold), because *any* long entry inherits
  the drift. We add a **return-shuffled placebo** that destroys the trend-vs-chop geometry the
  CI reads while keeping the price marginal — the direct test of "does the CI's structure
  matter?"

## Why the high one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample
  *t* of a long-only entry rule against **zero** measures that drift, not the rule. See Fama &
  French on the equity premium; the desk's standing rule is *excess-vs-excess* and
  *signal-vs-baseline*, never *signal-vs-zero*.
- **Data snooping on chart tools.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis*, Journal of Finance) formalize testing chart/indicator signals against a properly
  matched null; Sullivan, Timmermann & White (1999, *Data-Snooping, Technical Trading Rule
  Performance, and the Bootstrap*, JF) and White (2000, *A Reality Check for Data Snooping*,
  Econometrica) show how trend-fitted rules manufacture significance unless raced against a fair
  benchmark.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the low-CI-vs-random difference. True range is Wilder (1978, *New Concepts
  in Technical Trading Systems*).

## Method lineage (the desk's shared engine)

- **Trailing Choppiness Index + low-CI onset.**
  [`strategy.choppiness_index`](../choppiness_index/strategy.py),
  [`strategy.low_ci_entries`](../choppiness_index/strategy.py) — the mechanical gauge with the
  trailing-only discipline baked in.
- **Forward-return + HAC t + random baseline.**
  [`strategy.forward_returns`](../choppiness_index/strategy.py),
  [`strategy.hac_t`](../choppiness_index/strategy.py),
  [`strategy.run_experiment`](../choppiness_index/strategy.py).
- **Geometry placebo.** [`strategy.shuffled_returns_placebo`](../choppiness_index/strategy.py) —
  recompute CI on a return-permuted surrogate; keep the marginal, destroy the trend-vs-chop
  structure.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../choppiness_index/data.py)
  plants the exact "low-CI ⇒ upward momentum" structure (knob `edge`); with `edge = 0` the
  detector must NOT manufacture significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced
  by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../450-andrews-pitchfork`](../450-andrews-pitchfork) — the same "the channel/regime
  reverts/continues price" folklore tested with the random-entry baseline idiom (this study
  reuses its engine shape).
- [`../103-adx`](../103-adx) and the broader technical-indicator zoo — trend/chop "regime
  filters" mostly land None × Mirage for the same reason: an indicator fitted to past price
  re-describes the trend rather than forecasting it.
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting) frame why a
  signal-vs-zero *t* is not enough; the Choppiness Index is a clean live example of a
  volatility-geometry gauge whose apparent edge is beta.
