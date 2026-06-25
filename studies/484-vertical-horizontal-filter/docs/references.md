# References & literature map — Study 484 (Vertical-Horizontal-Filter)

## The claim under test

- **The folklore.** The **Vertical Horizontal Filter (VHF)** is a *regime classifier*: it tells
  you whether the market is **trending** or **ranging**, so you can switch your tools accordingly.
  The pitch, repeated on every indicator site (Investopedia, StockCharts, Incredible Charts,
  TradingView), is: *use trend-following / momentum / breakout systems only when the VHF is high
  (trending), and switch to oscillators / mean-reversion when the VHF is low (ranging).* Here we
  test the cleanest version of that: **gate a momentum entry on a high VHF** and ask whether the
  gate adds anything over the same momentum entry ungated.
- **The source.** **Adam White** introduced the VHF in *Futures* magazine (**"Filtering Out the
  Noise"**, August 1991). The formula is

      VHF_N = |highest_close(N) − lowest_close(N)| / Σ |close_t − close_{t−1}|   (last N bars).

  The numerator is the net **vertical** travel; the denominator is the total **horizontal** path
  length. VHF ∈ (0, 1]: near 1 ⇒ a clean directional move; near 0 ⇒ churn with little net
  progress. White's claim is that a *rising* VHF signals a trend worth following.
- **Siblings.** The VHF is one of a family of "trendiness" gauges — Wilder's **ADX** (Study 183),
  the **Choppiness Index**, **Kaufman's Efficiency Ratio** (the ratio at the heart of KAMA, Study
  433), and **R²/Hurst** trend filters (Study 392). They are affine/algebraic cousins: all
  normalize net displacement by path length, and all inherit the same confound tested here.

## Why this is a mechanical-proxy study

The VHF "use it to switch systems" rule is *semi-subjective* (what counts as "high"? which system
does it gate?). Following the desk's design, we encode the **tightest mechanical rule a proponent
would accept** and state the choices explicitly:

- **Objective momentum trigger.** Close above its 50-day moving average — the simplest, most-cited
  trend-following entry. Read on the close of *t*, no look-ahead.
- **Objective gate.** VHF (window 28) in the **top tertile** of its own trailing 252-day
  distribution — a causal "VHF says trending" threshold (no future data in the quantile).
- **The honest baseline.** The decisive comparison is **gated vs ungated** — both sides ride the
  same index drift, so the difference is drift-free *by construction*. We add a **drift-matched
  random-entry** control and a **shuffled-gate placebo** (permute the VHF in time, keep its
  marginal) — the direct test of "is the gate's *timing* doing anything?"

## Why the high one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample *t* of
  a long-only entry rule against **zero** measures that drift, not the rule. See Fama & French on
  the equity premium; the desk's standing rule is *signal-vs-baseline*, never *signal-vs-zero*. The
  gated momentum entry's big one-sample *t*'s (20d +3.35, 60d +5.97) are the *ungated* momentum
  entry's drift, unchanged.
- **Data snooping on indicators.** Lo, Mamaysky & Wang (2000, *Foundations of Technical Analysis*,
  *Journal of Finance*) formalize testing chart/indicator rules against a properly matched null;
  Sullivan, Timmermann & White (1999, *Data-Snooping, Technical Trading Rule Performance, and the
  Bootstrap*, JF) and White (2000, *A Reality Check for Data Snooping*, *Econometrica*) show how
  parameter-rich technical rules manufacture significance unless raced against a fair benchmark. A
  regime *gate* multiplies the free parameters (window, tertile, which system) — exactly the
  snooping surface those papers warn about.
- **Regime-switching skepticism.** The premise that you can classify "trend vs range" *in advance*
  and switch profitably is the harder claim; the academic record on real-time regime timing (e.g.
  the difficulty of out-of-sample market-timing, Welch & Goyal 2008, *RFS*) is discouraging.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the gate-vs-ungated and gate-vs-random differences.

## Method lineage (the desk's shared engine)

- **VHF indicator + momentum trigger.** [`strategy.vhf`](../vertical_horizontal_filter/strategy.py),
  [`strategy.momentum_signal`](../vertical_horizontal_filter/strategy.py).
- **Gated / ungated / random entries.** [`strategy.gated_entries`](../vertical_horizontal_filter/strategy.py),
  [`strategy.momentum_entries`](../vertical_horizontal_filter/strategy.py),
  [`strategy.random_entries`](../vertical_horizontal_filter/strategy.py).
- **Forward-return + HAC t + orchestrator.** [`strategy.forward_returns`](../vertical_horizontal_filter/strategy.py),
  [`strategy.hac_t`](../vertical_horizontal_filter/strategy.py),
  [`strategy.run_experiment`](../vertical_horizontal_filter/strategy.py).
- **Gate-timing placebo.** [`strategy.shuffled_gate_placebo`](../vertical_horizontal_filter/strategy.py) —
  permute the VHF in time, keep the marginal.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../vertical_horizontal_filter/data.py)
  plants a real VHF-conditional regime (knob `edge`); with `edge = 0` the gate must NOT beat
  ungated — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../../183-adx`](../../183-adx) — Wilder's ADX trend-strength gauge, the closest sibling regime
  filter; lands None × Mirage for the same drift reason.
- [`../../433-kama`](../../433-kama) — Kaufman's Efficiency Ratio (net change / path length) is the
  VHF's algebraic twin inside an adaptive moving average.
- [`../../392-hurst-exponent`](../../392-hurst-exponent) — the "is this series trending or
  mean-reverting?" question posed via R/S; a regime classifier with the same confound.
- The **research-method demos** (data-mining-roulette, multiple-testing, curve-fitting) frame why a
  signal-vs-zero *t* is not enough; a regime gate is a clean live example of a free parameter that
  re-describes the trend without forecasting it.
