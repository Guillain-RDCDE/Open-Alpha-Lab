# References & literature map — Study 497 (Woodie's Pivot Points)

## The claim under test

- **The folklore.** Pivot points are floor-trader levels: a central pivot P with support/
  resistance lines (S1, S2, R1, R2) derived from the prior session's range. The claim is that
  *yesterday's pivots act as today's intraday support and resistance* — price reaching down to
  **S1** finds support and bounces, reaching up to R1 meets resistance and fades. It is a retail
  and day-trading staple, plotted automatically by TradingView, MetaTrader, Thinkorswim and
  every charting suite.
- **The Woodie variant.** Where the classic floor-trader pivot is P = (H + L + C)/3, **Woodie's
  pivot double-weights the close**: P = (H + L + 2C)/4, with R1 = 2P − L, **S1 = 2P − H**,
  R2 = P + (H − L), S2 = P − (H − L). The close-weighting is attributed to **Ken Wood**
  ("Woodie"), founder of *Woodie's CCI Club*, an online day-trading community; his pivots and
  his Woodie-CCI system are the primary lineage. Popular write-ups (Investopedia's "Pivot
  Points" entry, John L. Person's *A Complete Guide to Technical Trading Tactics* (2004), and
  StockCharts' ChartSchool) restate the rule.
- **Variants.** Standard (floor-trader), Camarilla, Fibonacci, DeMark and Woodie pivots are all
  **affine functions of the same prior-day (H, L, C)** — they differ only in the weights/
  multipliers. They inherit the same drift confound tested here; Woodie's only distinction is
  the 2× close weight, which is precisely what the random-level placebo destroys.

## Why this is a mechanical-rule study

A pivot level is fully objective once you fix the formula and the lag — there is no eyeballing,
which makes Woodie's pivots an *unusually clean* mechanical test (unlike, say, a hand-anchored
pitchfork). The only design choices are:

- **Levels from the prior bar.** Woodie's P/S1/… for day *t* use the bar at *t−1*, knowable at
  *t*'s open — a documented one-bar lag, no look-ahead.
- **The honest baseline.** The only meaningful comparison on an upward-drifting index is the
  **random-entry** control (same instrument, epoch and hold), because *any* long entry inherits
  the drift — and an S1-touch fires on a majority of sessions, so it is nearly "always long".
- **The level placebo.** A **random-level** support (a re-sampled depth below the prior close)
  keeps the touch frequency and the marginal but destroys the close-weighting geometry — the
  direct test of "does the *specific* Woodie S1 matter?"

## Why the high one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample
  *t* of a long-only entry rule against **zero** measures that drift, not the rule. See Fama &
  French on the equity premium; the desk's standing rule is *excess-vs-excess* and
  *signal-vs-baseline*, never *signal-vs-zero*. This bites especially hard here because the
  S1-touch fires on most sessions — it is close to a buy-and-hold proxy.
- **Data snooping on chart tools.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis*, JF) formalize testing chart patterns against a properly matched null; Sullivan,
  Timmermann & White (1999, *Data-Snooping, Technical Trading Rule Performance, and the
  Bootstrap*, JF) and White (2000, *A Reality Check for Data Snooping*, Econometrica) show how
  price-fitted rules manufacture significance unless raced against a fair benchmark.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the touch-vs-random difference.

## Method lineage (the desk's shared engine)

- **Woodie levels.** [`strategy.woodie_levels`](../woodie_pivots/strategy.py) — P = (H+L+2C)/4
  and S1/S2/R1/R2 from the prior bar (the one-bar lag baked in).
- **S1-touch entry + forward-return + HAC t + random baseline.**
  [`strategy.s1_touch_entries`](../woodie_pivots/strategy.py),
  [`strategy.forward_returns`](../woodie_pivots/strategy.py),
  [`strategy.hac_t`](../woodie_pivots/strategy.py),
  [`strategy.run_experiment`](../woodie_pivots/strategy.py).
- **Level placebo.** [`strategy.random_level_placebo`](../woodie_pivots/strategy.py) — resample
  the depth-below-close, keep touch frequency and marginal, destroy the geometry.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../woodie_pivots/data.py)
  plants a real S1 support bounce (knob `edge`); with `edge = 0` the detector must NOT
  manufacture significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) OHLC for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced
  by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../../450-andrews-pitchfork`](../../450-andrews-pitchfork) — the sibling "price respects the
  drawn lines" study (median-line fork) with the same random-entry + geometry-placebo idiom.
- [`../../440-pivot-points`](../../440-pivot-points) and [`../../441-camarilla-pivots`](../../441-camarilla-pivots)
  — the classic and Camarilla pivot variants; Woodie is the close-weighted cousin and lands in
  the same place.
- [`../../104-bollinger-reversion`](../../104-bollinger-reversion) — the "the band/level reverts
  price" folklore tested with the random-entry baseline.
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting) frame why a
  signal-vs-zero *t* is not enough; Woodie's S1 is a clean live example of beta masquerading as a
  support level.
