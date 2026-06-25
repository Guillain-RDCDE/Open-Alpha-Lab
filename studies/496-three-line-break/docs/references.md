# References & literature map — Study 496 (Three-Line-Break)

## The claim under test

- **The folklore.** A *Three-Line-Break* (TLB) chart ignores time and volume and draws a new
  **line** (block) only when the close pushes past the prior line's extreme. The chart
  **reverses** colour only after the close breaks the extremes of the **3** most-recent
  opposite lines — hence "Three"-Line-Break. The lore: a TLB reversal **forecasts a new trend**,
  so you go **long on an up-line** and **flat (or short) on a reversal**, capturing trends while
  filtering minor noise. Built into TradingView, MetaStock, StockCharts and most charting suites.
- **The source.** TLB descends from the Japanese *Sakata* charting tradition (the same lineage
  as candlesticks and Renko). It was introduced to Western traders by **Steve Nison** in
  *Beyond Candlesticks* (1994), and codified as an indicator by **Steven B. Achelis** in
  *Technical Analysis from A to Z* (1995). The "break number" (default 3) is the only parameter;
  proponents tune it to trade off responsiveness against whipsaws.
- **Variants.** Two-Line-Break and Five-Line-Break (smaller/larger break numbers), and the
  closely related **Renko** and **Kagi** time-independent charts are affine cousins of the same
  "redraw on a close past a threshold" idea and inherit the same drift confound tested here.

## Why this is a "theory" / mechanical-proxy study

TLB is fully mechanical once the break number is fixed, but the *trading rule* attached to it
(long on up-lines, flat on reversals) is the discretionary folklore. Following the desk's design
for chart-tool claims, we encode the **tightest mechanical rule a proponent would accept** and
state the irreducible choices explicitly:

- **Causal construction.** TLB lines are built bar-by-bar from *past* closes only; a reversal is
  a function of the prior lines' extremes, so nothing leaks from the future. The reversal is read
  on the close of *t*; the trade is entered at the close of *t+1* (one documented lag).
- **Objective break number.** The default **3** lines, the value the name enshrines and every
  package ships.
- **The honest baseline.** The only meaningful comparison on an upward-drifting index is the
  **random-entry** control (same instrument, epoch and hold), because *any* long-only entry
  inherits the drift. We add a **shuffled-returns placebo** that destroys the specific line-break
  *sequence* while keeping the price marginal — the direct test of "does the break geometry
  matter?"

## Why the high one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample
  *t* of a long-only entry rule against **zero** measures that drift, not the rule. See Fama &
  French on the equity premium; the desk's standing rule is *excess-vs-excess* and
  *signal-vs-baseline*, never *signal-vs-zero*. Here the one-sample *t*'s reach +5.29/+6.26 yet
  the reversal *loses* to a random-day entry.
- **Data snooping on chart tools.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis*, Journal of Finance) formalize testing chart patterns against a properly matched
  null; Sullivan, Timmermann & White (1999, *Data-Snooping, Technical Trading Rule Performance,
  and the Bootstrap*, Journal of Finance) and White (2000, *A Reality Check for Data Snooping*,
  Econometrica) show how trend-fitted rules manufacture significance unless raced against a fair
  benchmark.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the reversal-vs-random difference.

## Method lineage (the desk's shared engine)

- **Causal TLB construction + reversal entries.**
  [`strategy.build_tlb`](../three_line_break/strategy.py),
  [`strategy.reversal_entries`](../three_line_break/strategy.py) — the mechanical line geometry
  with the entry lag baked in.
- **Forward-return + HAC t + random baseline.**
  [`strategy.forward_returns`](../three_line_break/strategy.py),
  [`strategy.hac_t`](../three_line_break/strategy.py),
  [`strategy.run_experiment`](../three_line_break/strategy.py).
- **Geometry placebo.** [`strategy.shuffled_returns_placebo`](../three_line_break/strategy.py) —
  permute the daily returns, keep the marginal, destroy the break sequence.
- **Deterministic synthetic control.**
  [`data.synthetic_panel`](../three_line_break/data.py) plants a real post-reversal continuation
  (knob `edge`); with `edge = 0` the detector must NOT manufacture significance — the offline
  core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced
  by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../../450-andrews-pitchfork`](../../450-andrews-pitchfork) — the sibling chart-geometry
  study this one is built from; same random-entry + geometry-placebo idiom.
- [`../../449-renko-charts`](../../449-renko-charts) and the broader time-independent-chart family (Renko,
  Kagi) — the closest cousins of TLB; same "redraw on a close past a threshold" mechanic.
- [`../../178-cci`](../../178-cci) and the broader technical-indicator zoo — most land
  None × Mirage for the same reason: an indicator fitted to past price re-describes the trend.
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting) frame why a
  signal-vs-zero *t* is not enough; TLB is a clean live example of beta masquerading as a
  reversal signal.
