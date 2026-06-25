# References & literature map — Study 463 (Bear-Flag)

## The claim under test

- **The folklore.** A **bear flag** is a continuation pattern: a sharp, near-vertical drop (the
  *pole* / *flagpole*) followed by a small *up-sloping* consolidation on lighter range (the
  *flag*, drifting counter-trend), and then a **breakdown** below the flag's lower edge that
  launches the *second leg down*. The classic target is a **measured move** — the breakdown
  point projected down by the height of the pole. The rule: *short the breakdown, the drop
  continues.* This is a retail/technician staple built into TradingView, StockCharts and every
  chart-pattern course.
- **The source.** Flags and pennants are catalogued in the founding technical-analysis texts —
  **Edwards & Magee**, *Technical Analysis of Stock Trends* (1948 and later editions) — and
  quantified, with explicit performance statistics, in **Thomas N. Bulkowski**, *Encyclopedia of
  Chart Patterns* (2nd ed., 2005). **John J. Murphy**, *Technical Analysis of the Financial
  Markets* (1999) restates the flag/pennant continuation rule and the measured-move target.
- **Variants.** Pennants (a small symmetric triangle instead of a parallel channel), bullish
  flags (the mirror image after an up-pole), and "high tight flags" are affine relatives of the
  same pole-then-consolidation geometry and inherit the same drift/short-side confounds tested
  here.

## Why this is a "theory" / mechanical-proxy study

A bear flag is *semi-subjective*: a discretionary trader decides which drop is a "pole" and
which sideways drift is a "flag". Following the desk's design for this kind, we encode the
**tightest mechanical rule a proponent would accept** and state the irreducible subjectivity
explicitly:

- **Objective pole.** A log-close fall of ≥ 6% over a 10-bar lookback (start-to-low) — a sharp,
  measurable drop, read only on bars up to *t* (no look-ahead).
- **Objective flag.** A 7-bar window with a *positive* OLS slope (up-sloping against the pole)
  whose total up-retrace stays under 60% of the pole (a pause, not a reversal); the lower
  trendline is the OLS fit shifted to the lowest residual.
- **Objective breakdown.** The first close below the extrapolated lower flag line; the short is
  entered at the **next close** (one documented lag).
- **The honest baseline.** The only meaningful comparison on a drifting index is the
  **random-entry** control on the *same short side* (same instrument, epoch and hold), because a
  short inherits the (negative) drift carry. We add a **shuffled-flag placebo** that replaces the
  up-sloping slope test with a coin while keeping the pole filter and the price marginal — the
  direct test of "does the flag's geometry matter?"

Hand-drawn flags add *hindsight* (which pole, which channel), a free parameter that can only
inflate in-sample fit; the mechanical version is therefore the charitable **upper bound** on the
method.

## Why the one-sample t is not evidence

- **Drift / beta (short side).** US equity indices have a positive unconditional daily mean, so
  a *short* held over any horizon has a built-in **negative** carry; a one-sample *t* of the
  breakdown-short against **zero** measures that drift, not continuation. See Fama & French on
  the equity premium; the desk's standing rule is *signal-vs-baseline*, never *signal-vs-zero*.
- **Data snooping on chart patterns.** **Lo, Mamaysky & Wang (2000)**, *Foundations of Technical
  Analysis: Computational Algorithms, Statistical Inference, and Empirical Implementation* (JF),
  formalize testing chart patterns against a properly matched null; **Sullivan, Timmermann &
  White (1999)**, *Data-Snooping, Technical Trading Rule Performance, and the Bootstrap* (JF),
  and **White (2000)**, *A Reality Check for Data Snooping* (Econometrica), show how rules fitted
  to past price manufacture significance unless raced against a fair benchmark.
- **Pattern catalogues over-report.** Bulkowski's own "success rate" tables are unconditioned on
  a drift-matched control and on multiple-testing across hundreds of patterns — the very inflation
  Sullivan-Timmermann-White warn about.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the breakdown-vs-random difference.

## Method lineage (the desk's shared engine)

- **Pole + up-sloping flag + breakdown.** [`strategy.detect_flags`](../bear_flag/strategy.py),
  [`strategy.flag_lower_line`](../bear_flag/strategy.py),
  [`strategy.breakdown_entries`](../bear_flag/strategy.py) — the mechanical geometry with the
  one-bar entry lag baked in.
- **Forward-return (short) + HAC t + random baseline.** [`strategy.forward_returns`](../bear_flag/strategy.py),
  [`strategy.hac_t`](../bear_flag/strategy.py), [`strategy.run_experiment`](../bear_flag/strategy.py).
- **Geometry placebo.** [`strategy.shuffled_flag_placebo`](../bear_flag/strategy.py) — replace
  the up-slope test with a coin, keep the pole and marginals.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../bear_flag/data.py) plants a
  real post-breakdown continuation (knob `edge`); with `edge = 0` the detector must NOT
  manufacture significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced
  by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../450-andrews-pitchfork`](../450-andrews-pitchfork) — the sibling "price respects the
  drawn lines" channel study; same random-entry-vs-geometry-placebo idiom, same None × Mirage.
- [`../410-head-and-shoulders`](../410-head-and-shoulders) and the broader chart-figure zoo —
  flags, triangles and reversal figures land None × Mirage for the same reason: a shape fitted
  to past price re-describes the move rather than forecasting the next one.
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting,
  multiple-testing) frame why a signal-vs-zero *t* is not enough; the bear flag is a clean live
  example of a famous pattern that, mechanized and raced against a fair short, forecasts nothing.
