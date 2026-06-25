# References & literature map — Study 467 (Bump-and-Run Reversal)

## The claim under test

- **The folklore.** A **bump-and-run reversal** (BARR) forms in three acts: (1) a gentle,
  low-slope **lead-in trendline** supporting a quiet advance; (2) a **bump** — a burst of
  speculation in which price *steepens* and surges far above the lead-in line (Bulkowski's rule
  of thumb: the bump rises to at least **2×** the lead-in's height above the line); (3) a
  **break** back below the lead-in trendline, which is taken as the **reversal signal**. The
  retail/technician rule is to **short** the trendline break (and a bullish mirror exists for
  bottoms). It is built into chart-pattern scanners and taught on every TA site.
- **The source.** **Thomas N. Bulkowski** named and catalogued the bump-and-run reversal in
  *Encyclopedia of Chart Patterns* (Wiley, 1st ed. 2000; 2nd ed. 2005) and on his site
  *thepatternsite.com*, reporting purported success rates and measured-move targets. The pattern
  descends from the broader trendline-break tradition (Edwards & Magee, *Technical Analysis of
  Stock Trends*, 1948) and the "speculative blow-off → reversal" lore.
- **Variants.** "Bump-and-run reversal bottoms", inverted BARRs, and the closely related
  "scallop"/"rounding" tops are affine relatives built on the same lead-in-line + over-extension
  + break geometry, and inherit the same drift/short-beta confound tested here.

## Why this is a "theory" / mechanical-proxy study

The BARR is *semi-subjective*: a discretionary chartist eyeballs where the lead-in line goes,
how steep counts as a "bump", and what counts as a clean break. Following the desk's design for
this kind, we encode the **tightest mechanical rule a proponent would accept** and state the
irreducible subjectivity explicitly:

- **Objective lead-in.** A least-squares trendline on a fixed trailing window, slope required to
  be *gently positive* (a calm up-trend, not a ramp), fit only on past bars — no look-ahead.
- **Objective bump.** The close must surge to ≥ `bump_mult`× the lead-in's own above-line height,
  and the bump peak must be *recent* (the rollover just happened), so stale geometry can't
  re-fire.
- **Objective break.** The first close that *downcrosses* the extended lead-in line; the short is
  entered at the **next close** (one documented lag).
- **The honest baseline.** The only meaningful comparison on an upward-drifting index is the
  **random-entry short** control (same instrument, epoch, hold, *and short side*), because *any*
  short inherits the negative drift. We add a **shuffled-window placebo** that permutes the
  per-bar returns — destroying the bump-and-run shape while keeping the price marginal — the
  direct test of "does the geometry matter?"

Hand-drawn BARRs add *hindsight* (which three swings, which slope), a free parameter that can
only inflate in-sample fit; the mechanical version here is the charitable **upper bound**.

## Why a high one-sample t is not evidence

- **Drift / short-beta.** US equity indices have a positive unconditional daily mean, so a
  one-sample *t* of a *short* rule against **zero** measures the index's drift (here, against the
  short), not the rule. The desk's standing discipline is *signal-vs-baseline*, never
  *signal-vs-zero* — the matched random short carries the identical drift headwind.
- **Data snooping on chart patterns.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis: Computational Algorithms, Statistical Inference, and Empirical Implementation*, JF)
  formalize testing chart patterns against a properly matched null; Sullivan, Timmermann & White
  (1999, *Data-Snooping, Technical Trading Rule Performance, and the Bootstrap*, JF) and White
  (2000, *A Reality Check for Data Snooping*, Econometrica) show how shape-fitted rules
  manufacture significance unless raced against a fair benchmark and a multiple-testing
  correction.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the break-vs-random difference.

## Method lineage (the desk's shared engine)

- **Trailing trendline + bump/break detection.** [`strategy.barr_break_entries`](../bump_and_run/strategy.py),
  [`strategy._fit_line`](../bump_and_run/strategy.py) — the mechanical geometry with the
  no-look-ahead lead-in and a recent-bump gate.
- **Forward-return (short side) + HAC t + random baseline.** [`strategy.forward_returns`](../bump_and_run/strategy.py),
  [`strategy.hac_t`](../bump_and_run/strategy.py), [`strategy.run_experiment`](../bump_and_run/strategy.py).
- **Geometry placebo.** [`strategy.shuffled_window_placebo`](../bump_and_run/strategy.py) —
  permute the per-bar returns, keep the marginal, destroy the ordering.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../bump_and_run/data.py)
  plants a real post-bump reversal (knob `edge`); with `edge = 0` the detector must NOT
  manufacture significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced
  by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../450-andrews-pitchfork`](../450-andrews-pitchfork) — the sibling channel/trendline-geometry
  study (same random-baseline + geometry-placebo idiom); also None × Mirage.
- [`../188-head-shoulders`](../188-head-shoulders) and [`../410-cup-and-handle`](../410-cup-and-handle)
  and the broader chart-pattern zoo — Bulkowski-style figures tested the same way mostly land
  None × Mirage: a shape fitted to past price re-describes the move it just had.
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting, multiple-testing)
  frame why a signal-vs-zero *t* is not evidence; the bump-and-run is a clean live example of a
  pattern whose break does not forecast the reversal it advertises.
