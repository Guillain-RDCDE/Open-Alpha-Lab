# References & literature map — Study 474 (Accelerator Oscillator)

## The claim under test

- **The folklore.** The **Accelerator Oscillator (AC)** measures the *acceleration* of momentum —
  the second derivative of price. Built on the Awesome Oscillator, AC = AO − SMA5(AO), where
  AO = SMA5(median price) − SMA34(median price). The lore: AC **leads** AO and price (acceleration
  changes direction *before* speed does), so two consecutive rising ("green") AC bars are a
  high-probability **buy**, strongest when AC is **above zero** ("never buy with a red bar"). It is a
  retail staple built into MetaTrader, TradingView, cTrader and every indicator suite.
- **The source.** **Bill Williams** introduced the Accelerator/Decelerator Oscillator (and the
  Awesome Oscillator, Alligator, Fractals, Gator) in *Trading Chaos* (1995) and *New Trading
  Dimensions* (1998, Wiley), as part of his "profitunity"/chaos-theory trading system. The AO/AC pair
  is his momentum-and-acceleration engine; the "two green bars above the zero line" entry is his own
  teaching, restated by Investopedia, BabyPips and the MetaTrader documentation.
- **Variants.** The Decelerator, the AO histogram crossing zero, and the AC "saucer" setups are
  affine reshapings of the same SMA-of-SMA arithmetic and inherit the same drift confound tested here.

## Why the high one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample *t* of a
  long-only entry rule against **zero** measures that drift, not the rule. The AC-up one-sample *t*'s
  here (20-day +6.50, 60-day +7.13) are large precisely because the rule fires often on an
  upward-drifting tape. See Fama & French on the equity premium; the desk's standing rule is
  *signal-vs-baseline*, never *signal-vs-zero*.
- **Data snooping on chart indicators.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis*, *Journal of Finance*) formalize testing chart/indicator patterns against a properly
  matched null; Sullivan, Timmermann & White (1999, *Data-Snooping, Technical Trading Rule
  Performance, and the Bootstrap*, *Journal of Finance*) and White (2000, *A Reality Check for Data
  Snooping*, *Econometrica*) show how trend-fitted rules manufacture significance unless raced against
  a fair benchmark. An SMA-of-SMA-of-SMA of past price is a textbook trend-fitted statistic.
- **Acceleration as a re-described derivative.** AC is a finite-difference second derivative of a
  smoothed price. On a drifting series with autocorrelated returns, *any* such derivative will look
  "predictive" of the trend it is computed from; the question is whether its *timing* relative to
  future price is load-bearing — which the rotated-AC placebo (p = 0.886) answers in the negative.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the AC-vs-random difference.

## Method lineage (the desk's shared engine)

- **Accelerator Oscillator + two-green-bars entry.**
  [`strategy.accelerator_oscillator`](../accelerator_oscillator/strategy.py),
  [`strategy.ac_entries`](../accelerator_oscillator/strategy.py) — the mechanical indicator with all
  windows trailing (no look-ahead) and entry at the next close.
- **Forward-return + HAC t + random baseline.**
  [`strategy.forward_returns`](../accelerator_oscillator/strategy.py),
  [`strategy.hac_t`](../accelerator_oscillator/strategy.py),
  [`strategy.run_experiment`](../accelerator_oscillator/strategy.py).
- **Timing placebo.** [`strategy.rotated_ac_placebo`](../accelerator_oscillator/strategy.py) —
  circularly rotate AC relative to price, keep the AC marginal exactly.
- **Deterministic synthetic control.**
  [`data.synthetic_panel`](../accelerator_oscillator/data.py) plants real upward-accelerating
  episodes (knob `edge`); with `edge = 0` the detector must NOT manufacture significance — the offline
  core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../../450-andrews-pitchfork`](../../450-andrews-pitchfork) — the gold-standard template: another
  Bill-Williams-adjacent chart tool that lands None × Mirage for the same drift reason.
- [`../../420-awesome-oscillator`](../../420-awesome-oscillator) — AC's parent indicator (the AO);
  AC is literally AO minus its own SMA5, so the two share a confound.
- [`../../178-cci`](../../178-cci) and the broader technical-indicator zoo — most land None × Mirage
  because an indicator fitted to past price re-describes the trend.
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting) frame why a
  signal-vs-zero *t* is not enough; the Accelerator Oscillator is a clean live example of a smoothed
  second derivative of price masquerading as a forecast.
