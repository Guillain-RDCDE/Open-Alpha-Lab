# References & literature map — Study 473 (Balance of Power)

## The claim under test

- **The folklore.** Balance of Power reads each bar as a tug-of-war between buyers and sellers,
  scaled by the bar's own range: **BOP = (close − open) / (high − low)**. A green bar closing
  near its high gives BOP ≈ +1 (buyers in control); a red bar near its low gives BOP ≈ −1
  (sellers in control). The raw series is noisy, so it is **smoothed** with a moving average.
  The claim, repeated across charting sites and trading forums, is that **smoothed BOP leads
  price** — when buyers gain the upper hand (smoothed BOP turns positive, crossing up through
  zero), a rise follows. We test that "buy the up-cross" rule.
- **The source.** **Igor Livshin** introduced Balance of Power in *Stocks & Commodities*
  ("Balance of Power", *Technical Analysis of Stocks & Commodities*, Aug. 2001) — note the same
  acronym BOP is sometimes used for an unrelated Worden/StockCharts indicator; this study
  encodes Livshin's body-over-range definition, the one built into many charting suites under
  his name. Livshin's pitch is that BOP measures the *strength* behind a move and so anticipates
  continuation.
- **Variants.** Smoothing length, EMA vs SMA, and "BOP histogram / signal-line cross" variants
  are affine tweaks of the same per-bar body-balance read-out and inherit the same drift
  confound tested here. Sibling oscillators — Chaikin Money Flow, Accumulation/Distribution,
  Elder's Force Index, the Klinger oscillator — all attempt the same "volume/body reveals who is
  in control" intuition.

## Why this is a "theory" / mechanical-proxy study

BOP itself is fully objective (a per-bar formula), but the *trading rule* a proponent draws from
it is not pinned down (which smoothing, which threshold, cross vs slope). Following the desk's
design, we encode the **tightest mechanical rule a proponent would accept** and state the choices
explicitly:

- **Objective indicator.** A causal trailing 14-day MA of BOP = (close−open)/(high−low), using
  only bars up to *t* — no look-ahead.
- **Objective entry.** A long on the **zero up-cross** (smoothed BOP negative on *t−1*,
  non-negative on *t*); enter the **next close** (one documented lag).
- **The honest baseline.** The only meaningful comparison on an upward-drifting index is the
  **random-entry** control (same instrument, epoch and hold), because *any* long-only entry
  inherits the drift. We add a **sign-scramble placebo** that permutes the per-bar BOP values
  before smoothing — destroying the temporal ordering the cross depends on while keeping the
  marginal — the direct test of "does the BOP sequence carry information?"

## Why the high one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample
  *t* of a long-only entry rule against **zero** measures that drift, not the rule. See Fama &
  French on the equity premium; the desk's standing rule is *signal-vs-baseline*, never
  *signal-vs-zero*. Here the one-sample *t* hits +6.13 at 60 days yet the cross *loses* to random
  — the whole apparent edge is beta.
- **Data snooping on chart tools.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis: Computational Algorithms, Statistical Inference, and Empirical Implementation*, JF)
  formalize testing chart-pattern rules against a properly matched null; Sullivan, Timmermann &
  White (1999, *Data-Snooping, Technical Trading Rule Performance, and the Bootstrap*, JF) and
  White (2000, *A Reality Check for Data Snooping*, Econometrica) show how price-fitted rules
  manufacture significance unless raced against a fair benchmark.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the cross-vs-random difference.

## Method lineage (the desk's shared engine)

- **Indicator + entries.** [`strategy.raw_bop`](../balance_of_power/strategy.py),
  [`strategy.smoothed_bop`](../balance_of_power/strategy.py),
  [`strategy.bop_cross_entries`](../balance_of_power/strategy.py) — the causal indicator and the
  zero up-cross rule.
- **Forward-return + HAC t + random baseline.** [`strategy.forward_returns`](../balance_of_power/strategy.py),
  [`strategy.hac_t`](../balance_of_power/strategy.py), [`strategy.run_experiment`](../balance_of_power/strategy.py).
- **Geometry placebo.** [`strategy.scramble_placebo`](../balance_of_power/strategy.py) — permute
  the per-bar BOP values, keep the marginal, destroy the ordering.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../balance_of_power/data.py)
  plants a real BOP-leads-price structure (knob `edge`); with `edge = 0` the detector must NOT
  manufacture significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced
  by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../../450-andrews-pitchfork`](../../450-andrews-pitchfork) — the sibling "the chart geometry
  forecasts" study whose engine (random baseline + scramble placebo + synthetic control) this
  one mirrors.
- [`../../419-chaikin-money-flow`](../../419-chaikin-money-flow) /
  [`../../423-force-index`](../../423-force-index) and the broader "who's in control" oscillator
  family — same body/volume intuition, same None × Mirage landing.
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting) frame why a
  signal-vs-zero *t* is not enough; Balance of Power is a clean live example of beta masquerading
  as a "buyer strength" signal.
