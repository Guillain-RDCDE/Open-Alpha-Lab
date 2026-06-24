# References & literature map — Study 450 (Andrews' Pitchfork)

## The claim under test

- **The folklore.** Draw a fork from three swing pivots: a *median line* from P0 through the
  midpoint of P1-P2, and two parallel *tines* through P1 and P2. "Price respects the fork" —
  it oscillates between the tines and is repeatedly drawn back to the median line, so a touch
  of the **lower tine** is a high-probability buy. This is the retail/technician staple built
  into TradingView, MetaTrader, Thinkorswim and every charting suite.
- **The source.** **Alan H. Andrews** developed the *median-line* (a.k.a. "pitchfork") method
  in the 1960s–70s, teaching it through his "Action–Reaction" course. His headline claim is
  that price reaches the median line roughly **80% of the time**. Roger Babson's
  action–reaction ideas and Andrews' own course notes are the primary lineage; the modern
  popular write-ups (Investopedia, *Technical Analysis of the Financial Markets* by John
  Murphy, and StockCharts' ChartSchool) restate the rule.
- **Variants.** Schiff and modified-Schiff forks shift the handle anchor; "warning lines" add
  further parallels. All are **affine variants of the same three-point geometry** and inherit
  the same drift confound tested here.

## Why this is a "theory" / mechanical-proxy study

Andrews' pitchfork is *semi-subjective*: a discretionary trader chooses which three swings to
anchor on. Following the desk's design for this kind, we encode the **tightest mechanical
rule a proponent would accept** and state the irreducible subjectivity explicitly:

- **Objective pivots.** Confirmed **fractals** (Bill Williams' fractal definition: a local
  extremum with *k* strictly-lower/higher bars on each side), only usable *k* bars later — a
  documented confirmation lag, no look-ahead.
- **Objective fork.** Anchored on the three most-recent confirmed pivots; no hand-picking.
- **The honest baseline.** The only meaningful comparison on an upward-drifting index is the
  **random-entry** control (same instrument, epoch and hold), because *any* dip-buy inherits
  the drift. We add a **shuffled-pivot placebo** that destroys the fork's geometry while
  keeping the price marginal — the direct test of "does the geometry matter?"

Hand-anchored forks add *hindsight* (a free parameter), which can only inflate in-sample fit;
the mechanical version is therefore the charitable **upper bound** on the method.

## Why the high one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample
  *t* of a long-only entry rule against **zero** measures that drift, not the rule. See Fama &
  French on the equity premium; the desk's standing rule is *excess-vs-excess* and
  *signal-vs-baseline*, never *signal-vs-zero*.
- **Data snooping on chart tools.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis*, JF) formalize testing chart patterns against a properly matched null; Sullivan,
  Timmermann & White (1999, *Data-Snooping, Technical Trading Rule Performance, and the
  Bootstrap*, JF) and White (2000, *A Reality Check for Data Snooping*, Econometrica) show how
  trend-fitted rules manufacture significance unless raced against a fair benchmark.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the touch-vs-random difference.

## Method lineage (the desk's shared engine)

- **Confirmed-fractal pivots + rolling fork.** [`strategy.find_pivots`](../andrews_pitchfork/strategy.py),
  [`strategy.build_forks`](../andrews_pitchfork/strategy.py) — the mechanical geometry with the
  confirmation lag baked in.
- **Forward-return + HAC t + random baseline.** [`strategy.forward_returns`](../andrews_pitchfork/strategy.py),
  [`strategy.hac_t`](../andrews_pitchfork/strategy.py), [`strategy.run_experiment`](../andrews_pitchfork/strategy.py).
- **Geometry placebo.** [`strategy.shuffled_pivot_placebo`](../andrews_pitchfork/strategy.py) —
  permute pivot prices, keep positions and marginals.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../andrews_pitchfork/data.py)
  plants a real tine-bounce (knob `edge`); with `edge = 0` the detector must NOT manufacture
  significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced
  by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../../104-bollinger-reversion`](../../104-bollinger-reversion) — the same "the band/channel
  reverts price" folklore tested with the random-entry baseline idiom.
- [`../../178-cci`](../../178-cci) and the broader technical-indicator zoo — most land
  None × Mirage for the same reason: an indicator fitted to past price re-describes the trend.
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting) frame why a
  signal-vs-zero *t* is not enough; the pitchfork is a clean live example of beta masquerading
  as a chart pattern.
