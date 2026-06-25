# References & literature map — Study 471 (QQE)

## The claim under test

- **The folklore.** The **QQE** (Quantitative Qualitative Estimation) indicator smooths an RSI
  and lays an ATR-of-RSI *trailing band* (the "fast" line) under/over it. The retail rule: a
  **long fires when the smoothed RSI crosses above the trailing band** — a "momentum ignition"
  that price continues; a cross below is a sell. This is a staple of the MetaTrader / TradingView
  / Forex-Factory crowd, with dozens of copy-paste implementations.
- **The source.** QQE was popularised by **Igor Livshin** (early-2000s forum/MetaTrader era) and
  is mechanically a re-packaging of **J. Welles Wilder Jr.'s** two canonical tools — the **RSI**
  and the **ATR** — from *New Concepts in Technical Trading Systems* (1978). The "4.236" Wilder
  factor (a Fibonacci-flavoured multiplier) and the Wilder smoothing (EMA with α = 1/length) come
  straight from that lineage. TradingView's built-in "QQE" / "QQE MOD" scripts (LazyBear, Mihkel00)
  are the modern reference implementations the rule here mirrors.
- **Variants.** "QQE MOD", "QQE with Bollinger", "Smoothed QQE" and the many coloured-histogram
  forks are **affine / cosmetic tweaks of the same smoothed-RSI-plus-trailing-band geometry** and
  inherit the same drift confound tested here.

## Why this is a "theory" / mechanical-proxy study

QQE is fully mechanical (no hand-drawing), so we encode the **tightest standard version a
proponent would accept** and state the parameter choices explicitly:

- **Causal construction.** RSI (Wilder, len 14), its EMA smoothing (sf 5), and the
  ATR-of-the-smoothed-RSI band (double Wilder smoothing × 4.236) are all one-sided recursions; the
  dual-band trailing stop flips between the long band (below) and short band (above) exactly as the
  reference scripts do. Nothing peeks forward.
- **The honest baseline.** The only meaningful comparison on an upward-drifting index is the
  **random-entry** control (same instrument, epoch and hold), because *any* long-only entry inherits
  the drift. Crucially the baseline is **pooled over many random seeds** — a single draw is too
  noisy and a lucky-low one fabricates a fake edge. We add a **phase-scramble placebo** (Fourier
  surrogate) that destroys the timing while keeping the spectrum/marginal — the direct test of "does
  the QQE geometry matter?"

## Why the high one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample *t*
  of a long-only entry rule against **zero** measures that drift, not the rule. See Fama & French
  on the equity premium; the desk's standing rule is *signal-vs-baseline*, never *signal-vs-zero*.
- **Baseline estimation error.** A single random-entry draw of a few hundred dates has a standard
  error on its mean of tens of bps; comparing the rule to *one* such draw is itself a coin-flip.
  Pooling seeds (the bootstrap idea) is what makes the comparator the true drift — and it is
  precisely what turns the apparent QQE "edge" back into nothing.
- **Data snooping on chart indicators.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis*, JF) formalize testing chart/indicator rules against a properly matched null; Sullivan,
  Timmermann & White (1999, *Data-Snooping, Technical Trading Rule Performance, and the Bootstrap*,
  JF) and White (2000, *A Reality Check for Data Snooping*, Econometrica) show how indicator rules
  manufacture significance unless raced against a fair benchmark.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the cross-vs-random difference.

## Method lineage (the desk's shared engine)

- **Causal QQE band + cross detection.** [`strategy.qqe_bands`](../qqe/strategy.py),
  [`strategy.qqe_cross_entries`](../qqe/strategy.py) — the dual-band trailing stop with the smoothing
  lag baked in.
- **Forward-return + HAC t + robust random baseline.** [`strategy.forward_returns`](../qqe/strategy.py),
  [`strategy.hac_t`](../qqe/strategy.py), [`strategy.random_baseline`](../qqe/strategy.py),
  [`strategy.run_experiment`](../qqe/strategy.py).
- **Geometry placebo.** [`strategy.phase_scramble_placebo`](../qqe/strategy.py) — Fourier
  phase-randomised surrogate, spectrum/marginal kept, timing destroyed.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../qqe/data.py) plants a real
  post-cross continuation (knob `edge`); with `edge = 0` the detector must NOT manufacture
  significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../450-andrews-pitchfork`](../450-andrews-pitchfork) — the same "geometry forecasts" folklore
  tested with the random-entry baseline + geometry placebo idiom.
- [`../178-cci`](../178-cci) and the broader technical-indicator zoo — most land None × Mirage for
  the same reason: an indicator fitted to past price re-describes the trend.
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting) frame why a
  signal-vs-zero *t* is not enough; QQE is a clean live example of beta masquerading as a momentum
  indicator — and a textbook case of an under-estimated baseline faking an edge.
