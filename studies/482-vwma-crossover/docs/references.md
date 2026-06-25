# References & literature map — Study 482 (VWMA-Crossover)

## The claim under test

- **The folklore.** A **volume-weighted moving average** weights each bar's price by its
  volume, VWMA_N = Σ(price·vol)/Σ(vol) over a trailing window. The retail/technician staple
  (built into TradingView, MetaTrader, Thinkorswim, NinjaTrader, and repeated across YouTube
  and chart-pattern sites) is that a VWMA crossover **front-runs** the plain simple-moving-average
  crossover: because the VWMA leans toward the bars where "real money" traded, a fast-above-slow
  VWMA *golden cross* is a higher-quality long trigger than the equal-weighted SMA golden cross.
- **The source.** The VWMA is a folk indicator with no single canonical inventor; it is the
  natural volume-weighted analogue of the simple moving average and a discrete cousin of the
  intraday **VWAP** (volume-weighted average price), whose institutional execution use is
  documented by Berkowitz, Logue & Noser (1988, *The Total Cost of Transactions on the NYSE*,
  JF) and Madhavan (2002, *VWAP Strategies*, in *Trading*). The moving-average **crossover**
  itself is the oldest mechanical trend rule — popularized by Donchian and tested at scale by
  Brock, Lakonishok & LeBaron (1992, *Simple Technical Trading Rules and the Stochastic
  Properties of Stock Returns*, JF). The volume-confirmation intuition traces to Granville's
  *On-Balance Volume* (1963) and the broader "volume precedes price" technician lore.
- **Variants.** EVWMA, VWAP-bands, and volume-weighted MACD are affine/parametric tweaks of the
  same volume-weighting idea and inherit the same drift confound tested here.

## Why this is a head-to-head incremental-value study

The honest question is **not** "does the VWMA cross make money?" (on an upward-drifting index,
*any* long-only golden cross does — that is drift). It is **does the volume term add anything
over the identical-length plain SMA cross?** So the design pins everything except the weighting:

- **Same fast/slow lengths (10/30), same golden-cross rule, same instrument, same epoch, same
  hold.** The only difference between the two legs is whether the window is volume-weighted.
- **The thesis test is VWMA − SMA** (a Welch two-sample *t*), not VWMA − zero.
- **The shuffled-volume placebo** permutes which volume attaches to which bar — keeping the
  price path and the volume marginal — so the weighting is destroyed while the marginals
  survive. The direct test of "is the volume term load-bearing?"

## Why the high one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample
  *t* of a long-only entry rule against **zero** measures that drift, not the rule. See Fama &
  French on the equity premium; the desk's standing rule is *signal-vs-baseline*, never
  *signal-vs-zero*.
- **Data snooping on chart tools.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis: Computational Algorithms, Statistical Inference, and Empirical Implementation*, JF)
  formalize testing chart/technical rules against a properly matched null; Sullivan, Timmermann
  & White (1999, *Data-Snooping, Technical Trading Rule Performance, and the Bootstrap*, JF) and
  White (2000, *A Reality Check for Data Snooping*, Econometrica) show how trend-fitted rules
  manufacture significance unless raced against a fair benchmark.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the VWMA-vs-SMA and VWMA-vs-random differences.

## Method lineage (the desk's shared engine)

- **Causal moving averages + golden-cross detection.** [`strategy.vwma`](../vwma_crossover/strategy.py),
  [`strategy.sma`](../vwma_crossover/strategy.py), [`strategy._golden_cross_dates`](../vwma_crossover/strategy.py)
  — trailing windows only; the cross is read on the close of *t*.
- **Forward-return + HAC t + random baseline.** [`strategy.forward_returns`](../vwma_crossover/strategy.py),
  [`strategy.hac_t`](../vwma_crossover/strategy.py), [`strategy.run_experiment`](../vwma_crossover/strategy.py).
- **Volume placebo.** [`strategy.shuffled_volume_placebo`](../vwma_crossover/strategy.py) —
  permute the volume series, keep the price path and the volume marginal.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../vwma_crossover/data.py)
  plants a real volume-led drift pulse (knob `edge`); with `edge = 0` the VWMA and SMA crosses
  are interchangeable — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes **and volume** for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced
  by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../../450-andrews-pitchfork`](../../450-andrews-pitchfork) — the same random-entry / placebo
  idiom on a charting tool; the template for this study.
- [`../../178-cci`](../../178-cci) and the broader technical-indicator zoo — most land
  None × Mirage for the same reason: an indicator fitted to past price re-describes the trend.
- [`../../119-obv`](../../119-obv) and other **volume-confirmation** rules — volume's repeated
  failure to add forecasting edge over price alone is a recurring desk finding.
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting) frame why a
  signal-vs-zero *t* is not enough; the VWMA cross is a clean live example of beta masquerading
  as a "smarter" moving average.
