# References & literature map — Study 686 (Stick Sandwich)

## The claim under test

- **The folklore.** The **stick sandwich** is one of Steve Nison's classic three-candle
  reversal patterns (*Japanese Candlestick Charting Techniques*, 1991 — the book that
  brought Japanese candle charting to Western technical analysis). The bullish version: a
  bearish candle, followed by a bullish candle that rallies *above* it, followed by a second
  bearish candle that gives the whole rally back and closes at **~the same price** as the
  first bearish candle. Visually two matching-close "bread" candles sandwich an up "filling"
  candle — hence the name. Nison's reading: the market tested the same price level twice
  (once from a decline, once from a failed rally) and it held both times, so it marks a
  **support level** worth buying. The pattern is explicitly framed as a *reversal*, so we
  steelman that with an explicit down-leg precondition rather than testing it in a vacuum.
- **Where it sits in the canon.** It is the three-candle, round-trip cousin of the two-candle
  **counterattack / meeting line** (sibling study
  [460-counterattack-lines](../460-counterattack-lines/)) — both patterns hinge on an
  *equal-close* condition as the load-bearing geometry; the stick sandwich adds a third bar
  and a completed round trip (down, up, back down to the same close) instead of a single
  gap-and-meet.
- **No dedicated academic literature.** Unlike morning/evening stars or engulfing patterns,
  the stick sandwich has essentially no independent academic testing — it survives almost
  entirely as chartist folklore repeated across trading-education sites. That absence of an
  academic anchor is itself informative: this is squarely the kind of claim the desk exists
  to test directly rather than take on faith.

## What we measure, and the honesty rails

- **Mechanical detection, no eyeballing.** All four prices of all three candles are known at
  the close of bar *t*; the sandwich condition (bearish / bullish-and-rallying / bearish,
  closes matching within 15 bps) plus a 10-day down-leg precondition are checked purely from
  OHLC. See [`stick_sandwich/strategy.py`](../stick_sandwich/strategy.py).
- **One documented execution lag.** The long is entered at the **next close** after the
  sandwich completes (one `shift`, applied once) — never the sandwich bar's own close.
- **The base rate, not zero, is the null.** A one-sample *t* against zero is a beta trap on
  an upward-drifting basket; the decisive statistic is the **Welch *t* of sandwich-minus-
  base-rate**, where the base rate is the identical long-only forward-return distribution
  measured on every eligible bar of the same panel.
- **Multiple-comparisons discipline.** Four horizons (5/10/20/60 days) are four simultaneous
  looks at the same question. We report a **Bonferroni-corrected** critical |*t*| for *k* = 4
  (≈ 2.50, via the two-sided normal quantile) alongside the naive |*t*| ≥ 2 bar, and neither
  is cleared at any horizon.
- **Geometry placebo.** The believers' mechanism is specifically the *equal close* — not
  merely "a down leg followed by a failed rally." We isolate that by drawing a same-size
  random sample from the pool of bars that share every condition **except** the equal-close
  test, and ask whether the real sandwiches beat that pool more than chance allows.
- **Costs charged one-way × NAV per leg** (5 bps, round trip = 10 bps/trade), long-only, no
  borrow (the claimed trade is a long).

## Data sources

- **SPY + 29 long-listed US large-cap daily OHLC**, auto-adjusted (total-return) — yfinance
  (no key), cached under `_cache/` (`sts_<TICKER>_1d.parquet`), 2001-07-10 → 2026-06-30. The
  basket is a 30-name slice of the desk's standard large-cap panel, also used by sibling
  study [685-tri-star-doji](../685-tri-star-doji/); survivorship (long-listed names) is
  named on the Signal axis — the base-rate control neutralizes it because it is measured on
  the identical survivor panel.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).
- Nison, Steve. *Japanese Candlestick Charting Techniques*, 2nd ed., New York Institute of
  Finance / Prentice Hall, 2001 — the canonical source for the stick-sandwich pattern name
  and its bullish/support reading.

## Related desk studies (the dedup map — what this study is NOT)

- [460-counterattack-lines](../460-counterattack-lines/) — the two-candle **meeting line**:
  a down candle then an up candle whose closes meet, no third bar, no round trip. Its
  verdict (also `None` × `Mirage`, equal-close placebo *p* = 0.623) is the closest sibling —
  same load-bearing mechanism (equal close), one fewer candle.
- [186-morning-star](../186-morning-star/) — a three-candle pattern, but the *middle* candle
  is the small-bodied "star" of indecision and the *third* candle is the one that
  penetrates back into the first candle's body; there is no equal-close requirement at all.
  Different geometry, different claim (indecision resolving directionally, not a tested
  support level).
- [452-spinning-top](../452-spinning-top/) — a single-candle indecision shape (small body,
  two long balanced wicks); no multi-candle structure, no equal-close condition.
- [459-hikkake-pattern](../459-hikkake-pattern/) — a false-breakout/inside-bar continuation
  pattern; shares "the tape fakes out and reverses" territory in spirit, but the detection
  geometry (inside bar + failed breakout) is unrelated to matching closes.
- [685-tri-star-doji](../685-tri-star-doji/) — three consecutive *doji* bars (tiny bodies);
  the stick sandwich instead requires two full-bodied *bearish* bars with matching closes
  around one full-bodied *bullish* bar — the opposite of "all three bars are indecisive."

None of the siblings test the stick sandwich's specific, defining claim: **that a rally
completely erased back to the same closing level, twice, inside three bars, calls a
bottom.** This study is the direct, honest test of that claim — and the geometry placebo is
what actually settles it (the equal close carries no shown information beyond the failed-rally
context).
