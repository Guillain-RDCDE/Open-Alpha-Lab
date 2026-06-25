# References & literature map — Study 475 (DeMarker)

## The claim under test

- **The folklore.** The DeMarker oscillator is bounded in [0, 1]; readings **above 0.7** flag
  overbought *exhaustion* and readings **below 0.3** flag oversold *exhaustion*. "Price exhausts
  and reverses at the extremes," so a DeMarker **rising up out of <0.3** is a high-probability buy
  (the down-move is "done"). This is the retail/technician staple built into MetaTrader,
  TradingView, Thinkorswim and most charting suites.
- **The source.** **Thomas R. DeMark** introduced the DeMarker (DeM) indicator in *The New
  Science of Technical Analysis* (Wiley, 1994), part of his family of exhaustion tools (TD
  Sequential, TD Combo, REI). The defining recursion: DeMax_t = max(High_t − High_{t-1}, 0),
  DeMin_t = max(Low_{t-1} − Low_t, 0), and DeMarker = SMA(DeMax, N) / (SMA(DeMax, N) + SMA(DeMin, N))
  over the classic N = 14. Unlike RSI it is built from the **highs and lows**, not the closes.
- **Variants.** Smoothed DeMarker (EMA instead of SMA of the DeMax/DeMin), different N, and
  alternative oversold/overbought thresholds (0.25/0.75, 0.30/0.70). All are monotone re-scalings
  of the same up-extension / (up+down) construction and inherit the same drift confound tested here.

## Why this is a mechanical-proxy study

DeMark's broader method (TD Sequential etc.) is *semi-subjective* — a discretionary trader reads
the oscillator alongside countdown counts and price flips. Following the desk's design for this
kind, we encode the **tightest mechanical rule a proponent would accept**:

- **Objective oscillator.** The DeMarker uses only bars through *t* — no look-ahead in its
  construction.
- **Objective trigger.** A long fires the first bar the DeMarker was below 0.3 yesterday and is
  higher today (turning *up* out of oversold); entry is the next close (one documented lag).
- **The honest baseline.** The only meaningful comparison on an upward-drifting index is the
  **random-entry** control (same instrument, epoch and hold), because *any* dip-buy inherits the
  drift. We add a **phase-scramble placebo** that rotates the DeMax/DeMin streams — destroying the
  oscillator's alignment with price while preserving its exact marginal — the direct test of "does
  the DeMarker's *timing* matter?"

## Why the high one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample *t* of
  a long-only entry rule against **zero** measures that drift, not the rule. See Fama & French on
  the equity premium; the desk's standing rule is *signal-vs-baseline*, never *signal-vs-zero*.
- **Data snooping on chart tools.** Lo, Mamaysky & Wang (2000, *Foundations of Technical Analysis:
  Computational Algorithms, Statistical Inference, and Empirical Implementation*, JF) formalize
  testing chart/indicator patterns against a properly matched null; Sullivan, Timmermann & White
  (1999, *Data-Snooping, Technical Trading Rule Performance, and the Bootstrap*, JF) and White
  (2000, *A Reality Check for Data Snooping*, Econometrica) show how an indicator with free
  parameters (period, thresholds, the choice of horizon) manufactures significance unless raced
  against a fair benchmark — directly relevant here, where significance appears at *one* of four
  horizons and is carried by *one* ticker.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the entry-vs-random difference.

## Method lineage (the desk's shared engine)

- **DeMarker oscillator + oversold-rising trigger.** [`strategy.demarker`](../demarker/strategy.py),
  [`strategy.oversold_rising_entries`](../demarker/strategy.py) — the mechanical indicator with no
  look-ahead.
- **Forward-return + HAC t + random baseline.** [`strategy.forward_returns`](../demarker/strategy.py),
  [`strategy.hac_t`](../demarker/strategy.py), [`strategy.run_experiment`](../demarker/strategy.py).
- **Timing placebo.** [`strategy.phase_scramble_placebo`](../demarker/strategy.py) — rotate the
  DeMax/DeMin streams, keep the oscillator marginal.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../demarker/data.py) plants a real
  exhaustion bounce keyed to the trigger (knob `edge`); with `edge = 0` the detector must NOT
  manufacture significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../450-andrews-pitchfork`](../450-andrews-pitchfork) — the same "channel/extreme reverts price"
  folklore, tested with the random-entry baseline and a geometry placebo (None × Mirage).
- [`../178-cci`](../178-cci), [`../180-stochastic`](../180-stochastic) and the broader
  oscillator/indicator zoo — most land None × Mirage; the DeMarker is one of the rare ones that
  shows a *fragile* signal at a single horizon rather than nothing at all.
- The **research-method demos** (data-mining-roulette, multiple-testing, curve-fitting) frame why a
  signal that appears at one of several horizons and is carried by one of several tickers should be
  graded Weak/Fragile, not Real — significance across many tried cuts is the classic snooping trap.
