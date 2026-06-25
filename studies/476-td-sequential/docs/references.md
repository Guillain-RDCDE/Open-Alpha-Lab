# References & literature map — Study 476 (TD Sequential — DeMark 9-13)

## The claim under test

- **The folklore.** A **TD Buy Setup** is nine consecutive closes, each strictly below the close
  four bars earlier; its completion (the "9") is the first exhaustion signal. A **TD Buy
  Countdown** then runs to thirteen "close ≤ low two bars earlier" rungs (the "13"), the deep
  exhaustion read. "Sellers are exhausted at the count" — so a completed setup/countdown is a
  high-probability **long**. This is the technician/trading-desk staple built into Bloomberg
  (DeMark studies), TradingView, Thinkorswim and most charting suites.
- **The source.** **Thomas R. DeMark** developed TD Sequential and published it in *The New
  Science of Technical Analysis* (Wiley, 1994); the indicator suite is licensed through Market
  Studies / DeMARK Analytics. Jason Perl's *DeMark Indicators* (Bloomberg Press, 2008) is the
  standard practitioner restatement of the 9-13 mechanics (setup, "perfection", countdown,
  recycling). The popular write-ups (Investopedia, the Bloomberg DeMark help pages, StockCharts
  ChartSchool) repeat the rule.
- **Variants.** "Setup perfection", TDST support/resistance, the TD Combo countdown, and the
  "aggressive" countdown are **parameter tweaks of the same close-vs-close-N / close-vs-low-M
  count** and inherit the same drift confound tested here. We encode the canonical 9 (lookback 4)
  and 13 (lookback 2).

## Why this is a mechanical-rule study

TD Sequential is *unusually objective* for technical analysis — the count is fully algorithmic —
so unlike a hand-drawn pattern there is **no eyeballing to remove**. We encode it verbatim
(9-consecutive close < close-4 setup; 13-rung close ≤ low-2 countdown; standard recycling on a
fresh setup) and state the irreducible test design:

- **No look-ahead.** Every rung uses only closes/lows at or before bar *t*; the completion is
  read on the close of *t* and the long is entered at the close of *t+1* (one documented lag).
- **The honest baseline.** The only meaningful comparison on an upward-drifting index is the
  **random-entry** control (same instrument, epoch and hold), because *any* dip-buy inherits the
  drift — and crucially we sample that baseline over **many seeds** (a single random draw can
  flatter or flatten the result by ±3 *t*-units). We add a **scrambled-lookback placebo** that
  replaces the canonical 4-bar comparison with other offsets, the direct test of "does the
  specific 9-13 count matter?"

## Why the high one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample *t*
  of a long-only entry rule against **zero** measures that drift, not the rule. See Fama & French
  on the equity premium; the desk's standing rule is *signal-vs-baseline*, never *signal-vs-zero*.
- **Data snooping on chart tools.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis*, Journal of Finance) formalize testing chart patterns against a properly matched
  null; Sullivan, Timmermann & White (1999, *Data-Snooping, Technical Trading Rule Performance,
  and the Bootstrap*, Journal of Finance) and White (2000, *A Reality Check for Data Snooping*,
  Econometrica) show how trend-fitted rules manufacture significance unless raced against a fair,
  resampled benchmark — exactly the seed-averaging that demotes this study's lucky single-seed
  Welch *t* from +2.91 to +2.07.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the setup-vs-random difference.

## Method lineage (the desk's shared engine)

- **TD setup / countdown.** [`strategy.buy_setup_count`](../td_sequential/strategy.py),
  [`strategy.buy_setup_entries`](../td_sequential/strategy.py),
  [`strategy.buy_countdown_entries`](../td_sequential/strategy.py) — the mechanical 9-13 count
  with the entry lag baked in.
- **Forward-return + HAC t + random baseline.** [`strategy.forward_returns`](../td_sequential/strategy.py),
  [`strategy.hac_t`](../td_sequential/strategy.py), [`strategy.run_experiment`](../td_sequential/strategy.py).
- **Geometry placebo.** [`strategy.scrambled_lookback_placebo`](../td_sequential/strategy.py) —
  permute the comparison offset, keep the setup length and marginal.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../td_sequential/data.py) plants
  a real post-setup exhaustion bounce (knob `edge`); with `edge = 0` the detector must NOT
  manufacture significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../../450-andrews-pitchfork`](../../450-andrews-pitchfork) — the same "the chart geometry
  reverts price" folklore, busted by a geometry-scramble placebo; the template for this study.
- [`../../104-bollinger-reversion`](../../104-bollinger-reversion) and the technical-indicator zoo
  ([`../../178-cci`](../../178-cci), [`../../182-stochastic`](../../182-stochastic) …) — most land
  None/Weak × Mirage for the same reason: an indicator fitted to past price re-describes the
  trend.
- The **research-method demos** ([`../../344-backtest-overfitting`](../../344-backtest-overfitting),
  [`../../346-multiple-testing`](../../346-multiple-testing)) frame why a single-seed *t* and a
  signal-vs-zero *t* are not enough; TD Sequential is a clean live example of a chart rule that
  *almost* clears the bar on a lucky seed and then evaporates under proper resampling.
