# References & literature map — Study 454 (Three-Outside-Up / -Down)

## The claim under test

- **The folklore.** The *three-outside* is a three-bar candlestick reversal. Bar 1 has a body of
  one colour; bar 2 is the opposite colour and its real body **engulfs** bar 1 (the classic
  bullish/bearish **engulfing** pattern); bar 3 **confirms** by closing further in the engulf
  direction. The teaching is that the confirmation upgrades the engulf into a high-probability
  reversal — buy a three-outside-**up**, sell a three-outside-**down**. It is a staple of every
  candlestick primer and is recognised by TA-Lib (`CDL3OUTSIDE`), TradingView, and most charting
  suites.
- **The source.** Japanese candlestick charting was introduced to the West by **Steve Nison**,
  *Japanese Candlestick Charting Techniques* (1991) and *Beyond Candlesticks* (1994), tracing the
  method to Munehisa Homma's 18th-century rice trading. The "outside" three-bar variants
  (three-outside-up/down) are catalogued there and in **Gregory Morris**, *Candlestick Charting
  Explained* (1992/2006). The engulfing core is the load-bearing two-bar piece; the third bar is
  Nison's "confirmation."
- **The evidence base.** **Thomas Bulkowski**, *Encyclopedia of Candlestick Charts* (2008),
  tabulates hit-rates for the three-outside-up/down and ranks them mid-pack; his "reversal rates"
  are computed *without* a drift-matched benchmark, which is exactly the confound this study
  isolates.

## Why this is a mechanical-proxy study

A discretionary candlestick reader allows wiggle in "engulf" and "confirm." We encode the
**tightest mechanical rule a proponent would accept** and state the geometry explicitly:

- **Objective engulf.** Real-body engulfing only (open/close coordinates), opposite colours, the
  second body fully covering the first — no shadow rules, no gap rules, the standard strict form.
- **Objective confirmation.** Bar 3 closes beyond bar 2's close in the engulf direction; the
  pattern is read on the close of the confirming bar (no look-ahead) and entered the next close.
- **The honest baseline.** The only meaningful comparison on an upward-drifting index is the
  **random-entry** control (same instrument, epoch and hold), because *any* long-only rule
  inherits the drift. We add a **confirmation-shuffle placebo** that pulls size-matched entries
  from the pool of *all* engulfs, ignoring the third bar — the direct test of "does the
  confirmation matter beyond the bare engulf?"

## Why the high one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample *t*
  of a long-only entry rule against **zero** measures that drift, not the rule. See Fama & French
  on the equity premium; the desk's standing rule is *signal-vs-baseline*, never *signal-vs-zero*.
- **Data snooping on chart patterns.** **Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis*, Journal of Finance)** formalise testing chart patterns (including candlesticks)
  against a properly matched null and find most add little once conditioned correctly.
  **Sullivan, Timmermann & White (1999, *Data-Snooping, Technical Trading Rule Performance, and
  the Bootstrap*, JF)** and **White (2000, *A Reality Check for Data Snooping*, Econometrica)**
  show how pattern-mined rules manufacture significance unless raced against a fair benchmark.
- **Candlesticks specifically.** **Marshall, Young & Rose (2006, *Candlestick technical trading
  strategies: Can they create value for investors?*, Journal of Banking & Finance)** apply a
  bootstrap to Dow stocks and find candlestick signals add no value — the same conclusion this
  study reaches for the three-outside-up.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the pattern-vs-random difference.

## Method lineage (the desk's shared engine)

- **Engulf + confirm detection.** [`strategy._engulf_flags`](../three_outside/strategy.py),
  [`strategy.three_outside`](../three_outside/strategy.py) — the mechanical two-bar engulf plus
  third-bar confirmation, read on the close of *t*.
- **Forward-return + HAC t + random baseline.** [`strategy.forward_returns`](../three_outside/strategy.py),
  [`strategy.hac_t`](../three_outside/strategy.py), [`strategy.run_experiment`](../three_outside/strategy.py).
- **Confirmation placebo.** [`strategy.confirm_shuffle_placebo`](../three_outside/strategy.py) —
  size-matched draws from the pool of all engulfs, confirmation ignored.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../three_outside/data.py) plants
  a real multi-day continuation after each confirmed pattern (knob `edge`); with `edge = 0` the
  detector must NOT manufacture significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../../450-andrews-pitchfork`](../../450-andrews-pitchfork) — the same drift/random-baseline
  idiom on a charting tool; same None × Mirage outcome.
- The candlestick zoo (engulfing, stars, soldiers, harami) and the broader technical-indicator
  studies — most land None × Mirage because the pattern re-describes the trend rather than
  forecasting it.
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting, multiple-testing)
  frame why a signal-vs-zero *t* is not enough; the three-outside-up is a clean live example of a
  "confirmed" pattern whose confirmation does nothing.
