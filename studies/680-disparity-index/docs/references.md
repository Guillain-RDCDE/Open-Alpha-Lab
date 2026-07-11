# References & literature map — Study 680 (Disparity Index)

## The claim under test

- **The folklore.** The **Disparity Index** — DI(N) = 100 x Close / SMA(N) — measures how
  far the close has stretched from its own trailing N-day average, expressed as a
  percentage. It is a staple of Korean and Japanese retail technical-analysis education
  (alongside the Psychological Line and the Ichimoku family): DI meaningfully above 100
  ("stretched up") signals overbought/sell, meaningfully below 100 ("stretched down")
  signals oversold/buy. Short-horizon practice typically pairs a 10-day window with a
  ±5% band — the exact defaults tested here.
- **The academic anchor is thin and indirect.** Moving-average-relative price gaps are
  a special case of the broad "distance-from-trend mean reversion" literature that traces
  to **DeBondt & Thaler (1985, *Does the Stock Market Overreact?*, JF)** and the
  short-horizon reversal documented by **Jegadeesh (1990, *Evidence of Predictable Behavior
  of Security Returns*, JF)** and **Lehmann (1990, *Fads, Martingales, and Market
  Efficiency*, QJE)** — the same anchor cited by sibling study
  [329-one-month-reversal](../../329-one-month-reversal/). There is no dedicated Western
  academic literature on the Disparity Index specifically; it is a technician's tool, not
  a factor with a published risk-premium story, and Wilder's / Bollinger's Western
  band-distance cousins (see the dedup map) have the same status.

## What we measure, and the honesty rails

- **DI computed causally.** DI(N) at bar *t* uses the SMA of the trailing N closes
  *including* bar *t*'s own close — known at the close, tradable at *t+1*'s open. One
  documented execution lag throughout (data.py / strategy.py docstrings).
- **Two complementary measurements share the indicator**: a conditional-forward-return
  split (bucket every day by its DI reading, Welch *t* + NW(*h*) cross-check for the
  overlapping-window autocorrelation) and a zone-**trigger** trade ledger (only the day
  DI first crosses into a zone fires a trade, avoiding stacking near-duplicate signals),
  pinned against a random-direction coin.
- **The decisive control is the random-DAY baseline**, not just the random-direction
  coin: the universe pools two-plus-decade bull-drift names (NVDA, TSLA), so "does DI beat
  a coin flip on the days it fires" is a weaker question than "does DI beat simply buying
  an arbitrary day of the same stock." The latter is what actually decides the verdict
  here — see [`docs/results.md`](results.md).
- **The "just reversal?" diagnostic** reports the pooled correlation between DI and the
  plain trailing N-day return directly (r = 0.84) — DI is a smoothed transform of the same
  underlying quantity, not an independent signal.
- **Parameter grid** sweeps window {5,10,20,25} x threshold {±3,±5,±7}% — the reported
  asymmetry (overbought leg positive everywhere) is structural, not a cherry-picked cell.

## Why the myth-check axis exists

The claim has two contrarian halves — buy the oversold, sell/short the overbought — and
this desk's job is to test both, not just the flattering one. On this real, drift-heavy
tape the overbought half is **backwards**: DI > 105 predicts *above-average* forward
returns. That is exactly the signature of momentum continuation in a small basket of
secular winners, not mean reversion, and it is the reason the literal two-sided rule
loses money net of costs (see the timer table in `results.md`).

## Data sources

- **Daily total-return-adjusted OHLC**, SPY/QQQ/IWM/AAPL/TSLA/NVDA — yfinance (no key),
  cached under `_cache/` (`di_<TICKER>.csv`), 2003-01-02 → 2026-06-30 (TSLA from its
  2010-06-29 IPO). Identical universe and cache shape to sibling
  [679-psychological-line](../../679-psychological-line/).
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [329-one-month-reversal](../../329-one-month-reversal/) — the direct ancestor. Ranks
  the cross-section by **last month's return** and finds a real (*t* = +2.40), but
  bid-ask-bounce-driven, dead-since-2002 reversal. DI is a smoothed version of exactly
  that trailing-return signal (r = 0.84 here) — same object, different label, same fate.
- [104-bollinger-reversion](../../104-bollinger-reversion/) — the Western cousin: price
  vs a band built from a moving average ± *σ*, instead of DI's moving average ± a fixed
  percentage. That study found the identical drift-contamination pattern (lower-band
  entries barely beat a random-day buy; the upper-band "opposite" rule also profits) —
  this study runs the equivalent random-day control and reaches the same structural
  conclusion independently.
- [137-mansfield-rs](../../137-mansfield-rs/) — relative *strength* vs a benchmark
  (trend-following, buy the leaders). The opposite trading philosophy to a
  disparity-index contrarian rule; tests whether trend-following timing beats a random
  entry (it does not, there either).
- [679-psychological-line](../../679-psychological-line/) — the nearest sibling in
  *design*: an oscillator (share of up-closes, not price-distance) tested with the exact
  same protocol shape (conditional split + zone-trigger ledger + random-direction
  control) on the identical six-ticker universe. Different indicator, same house
  methodology — the two studies are a matched pair, not duplicates.

None of the siblings run the **DI-specific price-distance-from-its-own-MA** rule with a
random-*day* drift control — that combination is this study's own contribution, and it is
what exposes the drift contamination the raw split alone would have missed.
