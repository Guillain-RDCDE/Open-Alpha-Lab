# References & literature map — Study 495 (Kagi Charts)

## The claim under test

- **The folklore.** A **Kagi chart** is a price-only Japanese chart that ignores time: it draws
  a vertical line in the current direction while price keeps moving that way, and **reverses**
  only when price moves against it by at least a fixed *reversal* amount. Its signature is line
  **thickness** — the line turns **yang** (thick) when a rising segment breaks the prior
  **shoulder** (the last swing high), and **yin** (thin) when a falling segment breaks the prior
  **waist** (the last swing low). The trading rule, taught everywhere: **buy when the line turns
  yang** (thick = demand in control, uptrend confirmed) and **go flat/sell on yin**.
- **The source.** The Kagi chart originated in 1870s Japan (around the opening of the Japanese
  stock market). It was introduced to Western traders by **Steve Nison** in *Beyond
  Candlesticks* (Wiley, 1994), which codified the yin/yang thickness rule and the shoulder/waist
  vocabulary. Modern restatements appear in StockCharts' ChartSchool, Investopedia, and the
  charting suites (TradingView, MetaTrader) that ship a Kagi overlay.
- **Variants.** The reversal can be set as a **fixed percentage** (used here, 4% default), a
  fixed point amount, or an **ATR multiple**; some implementations use the close, others the
  high/low. All are **monotone re-parameterisations of the same shoulder/waist geometry** and
  inherit the same drift confound tested here — which is exactly what the threshold-scramble
  placebo demonstrates.

## Why this is a "theory" / mechanical-proxy study

The Kagi chart is *semi-objective*: the line itself is mechanical once the reversal is fixed,
but a discretionary trader chooses the reversal size and may read the yin/yang "feel". Following
the desk's design for this kind, we encode the **tightest mechanical rule a proponent would
accept** and state the irreducible parameter choice explicitly:

- **Objective line.** Built left-to-right from closes only; a reversal of 4% of the turning
  price flips the direction and records a shoulder/waist. No future bars enter the line.
- **Objective switch.** The yang switch is the bar the line first trades above the recorded
  shoulder (thin→thick); the yin switch the bar it first trades below the waist. Read on the
  close of *t*, entered at the close of *t+1*.
- **The honest baseline.** The only meaningful comparison on an upward-drifting index is the
  **random-entry** control (same instrument, epoch and hold), because *any* long-only rule
  inherits the drift. We add a **threshold-scramble placebo** that rebuilds the Kagi with a
  randomly-drawn reversal (1%–8%), keeping the price marginal — the direct test of "does *this*
  Kagi geometry matter?"

A trader who hand-tunes the reversal to the chart adds *hindsight* (a free parameter), which can
only inflate in-sample fit; the fixed-4% mechanical version here is the charitable **upper
bound** on the method.

## Why the one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample
  *t* of a long-only entry rule against **zero** measures that drift, not the rule. See Fama &
  French on the equity premium; the desk's standing rule is *excess-vs-excess* and
  *signal-vs-baseline*, never *signal-vs-zero*. Here even the 60-day one-sample *t* (+2.84)
  evaporates to a Welch +1.27 against random entries.
- **Data snooping on chart tools.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis: Computational Algorithms, Statistical Inference, and Empirical Implementation*, JF)
  formalize testing chart patterns against a properly matched null; Sullivan, Timmermann & White
  (1999, *Data-Snooping, Technical Trading Rule Performance, and the Bootstrap*, JF) and White
  (2000, *A Reality Check for Data Snooping*, Econometrica) show how trend-fitted rules
  manufacture significance unless raced against a fair benchmark and corrected for the universe
  of rules tried.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the switch-vs-random difference.

## Method lineage (the desk's shared engine)

- **Mechanical Kagi line + yang/yin switch.** [`strategy.kagi_line`](../kagi_charts/strategy.py),
  [`strategy.yang_switch_entries`](../kagi_charts/strategy.py) — the line geometry with the
  reversal threshold and shoulder/waist thickness baked in, all causal in *t*.
- **Forward-return + HAC t + random baseline.** [`strategy.forward_returns`](../kagi_charts/strategy.py),
  [`strategy.hac_t`](../kagi_charts/strategy.py), [`strategy.run_experiment`](../kagi_charts/strategy.py).
- **Geometry placebo.** [`strategy.threshold_scramble_placebo`](../kagi_charts/strategy.py) —
  re-parameterise the reversal, keep the marginal.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../kagi_charts/data.py) plants a
  real post-yang-switch momentum burst (knob `edge`); with `edge = 0` the detector must NOT
  manufacture significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced
  by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../450-andrews-pitchfork`](../450-andrews-pitchfork) — the sibling "price respects the
  channel" chart tool, same random-entry + geometry-placebo idiom; also None × Mirage.
- [`../178-cci`](../178-cci) and the broader technical-indicator zoo — most land None × Mirage
  for the same reason: an indicator fitted to past price re-describes the trend.
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting) frame why a
  signal-vs-zero *t* is not enough; the Kagi yang switch is a clean live example of beta
  masquerading as a chart pattern.
