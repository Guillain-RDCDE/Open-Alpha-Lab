# References & literature map — Study 692 (Breakaway Candles)

## The claim under test

- **The folklore.** The **breakaway** is a five-candle reversal from the Japanese
  candlestick canon: candle 1 is a long-bodied candle continuing the prevailing trend;
  candle 2 **gaps away** from it (a genuine window that stays open); candles 3-4 **run**
  further in that direction; candle 5 is a long-bodied candle in the *opposite* direction
  that closes back **through the gap**. Nison's own description is that the fifth candle
  "closes within the area of the window" — it erases the last leg of the run. The claim:
  this exact five-bar shape marks the end of the trend it interrupted, in *either*
  direction — a **bullish breakaway** ends a downtrend, a **bearish breakaway** ends an
  uptrend.
- **The academic anchor.** Steve Nison, *Japanese Candlestick Charting Techniques* (2nd
  ed., 2001), and Thomas Bulkowski, *Encyclopedia of Candlestick Charts* (2008) — both
  catalog the breakaway among the classic reversal shapes; Bulkowski's own event-study
  performance tables (US equities, pre-2008) are the closest prior empirical read, and
  they already flag it as one of the *weaker* five-candle reversal patterns by his own
  ranking — a useful prior against inflating expectations here.
- **The mechanism claimed.** A gap that *doesn't* fill for several sessions is read as
  conviction; a long reversal candle that erases it in one session is read as an
  exhaustion signal — the same "the crowd was wrong, and now it knows it" logic behind
  every gap-cluster reversal figure on the desk.

## What we measure, and the honesty rails

- **Two cuts, loose and strict**, exactly the loose/strict idiom used by sibling studies
  685-tri-star-doji and 687-ladder-bottom: the loose cut is the plain mechanical reading
  (a clean gap, a monotonic run, a same/opposite-direction close crossing back through
  the gap-day's own high/low); the strict cut additionally requires a **bigger** gap,
  genuinely **long** bodies on candles 1 and 5 (Nison's own emphasis), and a **full**
  gap-fill (the reversal closes back past candle 1's own extreme, not just candle 2's).
- **The base rate.** Every reversal trade is compared against the *unconditional* base
  rate: the same directional bet on **every** bar that also sits in a matching
  downtrend/uptrend context, whether or not the specific five-candle shape fired. This is
  the same discipline as 687 — it isolates "does the breakaway shape add information
  beyond simply being in a trend" from plain trend-context mean reversion.
- **Bidirectional, pooled AND split.** Unlike a single-direction figure, the breakaway is
  claimed both ways, so the headline test **pools** bullish and bearish events (both
  already sign-adjusted to "trade P&L"); the two sides are also reported **separately**
  as the desk's own symmetry myth-check, the same idea 417-island-reversal uses for its
  island-top/island-bottom split.
- **Four horizons -> Bonferroni.** 1/5/10/20-day forward windows means a
  Bonferroni-corrected critical value (k=4), not a naive \|t\| >= 2 — the same discipline
  687 applies.
- **The honest sample-size rule.** Below `MIN_N_FOR_TEST` (8) pooled events, no *t*-stat
  is computed at all — a *t* on a handful of points is decoration, not evidence.
- **A label-shuffle placebo** draws the same number of "fake" events from the pooled
  base-rate pool and asks how often a random draw beats the observed mean — the honest
  control for the basket's own drift, the same idiom 687 uses.

## Data sources

- **Daily OHLCV**, SPY + 60 long-listed US large-caps — yfinance (no key), cached under
  `_cache/` (`bwc_<ticker>_1d.parquet`), ~25 years each, as-of **2026-06-30**. This is the
  **same fixed basket** used by sibling studies 685-tri-star-doji / 687-ladder-bottom — a
  rare multi-bar candle shape needs a broad, long-listed basket for any usable sample at
  all. **Survivors basket** (all still trading) — named on the Signal axis, though for a
  single-pattern event study it affects which *names* contribute events, not the
  direction of the comparison.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [417-island-reversal](../417-island-reversal/) — a **two-gap** figure (exhaustion gap,
  a stranded island of bars, a sealing gap the *opposite* way) with **no run** in
  between; that study also splits top/bottom the same way, but the breakaway is a
  **one-gap-then-run-then-reversal** shape, structurally different from an island's
  bracket-and-strand.
- [74-mind-the-gap](../74-mind-the-gap/) — tests whether **any single opening gap**
  fills, with no candle-count structure at all (fade-the-gap, one bar). The breakaway
  is a specific **five-bar** sequence around a gap that is explicitly *not* supposed to
  fill until the reversal candle — the opposite emphasis.
- [455-three-methods](../455-three-methods/) — a different five-candle shape (the
  "rising/falling three methods"), a **continuation** pause inside an ongoing trend, not
  a reversal, and it never involves a gap at all.
- [687-ladder-bottom](../687-ladder-bottom/) — the desk's closest structural cousin: a
  five-candle, loose/strict, base-rate-vs-Bonferroni idiom this study reuses directly.
  But the ladder bottom is a **no-gap**, single-direction (bullish only) figure built
  from four *declining* candles and a break; the breakaway is bidirectional and its
  entire premise is the **gap that holds and then gets erased**.

None of the siblings test the specific **gap, then a run, then a reversal candle back
through the gap** five-bar shape in *both* directions — the breakaway is this study's own
axis.
