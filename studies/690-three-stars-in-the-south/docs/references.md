# References & literature map — Study 690 (Three Stars in the South)

## The claim under test

- **The folklore.** **Three stars in the south** is one of Japanese candlestick lore's
  more obscure bullish reversal claims: **Steve Nison**, *Japanese Candlestick Charting
  Techniques* (1991, 2nd ed. 2001) — describes a downtrend producing three consecutive
  **black** candles, each with a visibly **smaller body/range** than the one before, and
  each showing a **higher low** than the one before — sellers still pushing, but with
  less and less conviction each session, until the third candle is a small,
  near-marubozu body that doesn't even test the second candle's low. The claim: the
  shrinking, rising-low sequence itself *is* the exhaustion signal, no confirming
  bullish candle required. **Thomas Bulkowski** (*Encyclopedia of Candlestick Charts*,
  2008) lists it among the rarest patterns in the canon and cautions that he could not
  find enough historical occurrences to rank its performance with any confidence — the
  honest starting point for this study.
- **The steelman.** *A three-stars-in-the-south block, entered the session after it
  confirms, earns a forward long return that beats the unconditional base rate of
  buying any bar that is already in a matching downtrend — net of costs — because the
  shrinking/rising-low geometry itself signals that sellers are running out of
  ammunition.*

## Why this is a mechanical-proxy study, and the two-cut design

Like its rarer candlestick cousins, three stars in the south is *semi-subjective*: a
chartist eyeballs "shrinking bodies" and "rising lows." Following the desk's design for
this kind of claim (see the sibling studies below), we encode the tightest mechanical
rule a proponent would accept and report **two cuts** side by side:

- **Loose.** Three consecutive bearish candles with strictly shrinking intrabar ranges
  and strictly rising lows, sitting in a genuine prior downtrend. Common enough to run
  inference on (n in the hundreds to low thousands, pooled).
- **Strict, literature-closer.** The loose cut, plus (a) the first star shows a real
  lower shadow (a hammer-like first candle — selling met by some intrabar buying), (b)
  the second star opens **inside** the first star's real body (no gap down to start the
  sequence), and (c) the third star is a **near-marubozu** (small shadows both sides —
  a decisively small, committed body) that never breaks below the second star's low.
  This is the primary claim under test, and — as with the desk's other rare
  candlestick patterns — it is genuinely uncommon.

## Why the honest sample-size rule and Bonferroni correction matter here

- **Four horizons, multiple looks.** We read forward 1/5/10/20-day returns — four
  simultaneous hypotheses. At the usual *α* = 5% level, roughly one spurious hit is
  expected every 20 independent tries; we apply a **Bonferroni correction** (k = 4,
  critical |*t*| ≥ **2.50**) and report which, if any, horizon survives it — the same
  discipline sibling studies 186 (morning-/evening-star), 685 (tri-star doji) and 687
  (ladder bottom) use for their own multi-horizon grids.
- **Below `MIN_N_FOR_TEST` (8 pooled events), no *t*-stat is computed at all** — a *t*
  on a handful of trades is decoration, not evidence. Both cuts' full event lists are
  reproducible from [`examples/verify.py`](../examples/verify.py).

## Why a high one-sample *t* would not be evidence even if it appeared

- **Drift / beta and downtrend mean reversion.** US equities carry positive
  unconditional drift, and a bar that is already in a downtrend has its own mean-
  reversion tendency independent of any three-candle shape. The desk's standing rule
  is *signal-vs-baseline*, never *signal-vs-zero* — the decisive number here is the
  three-stars reversal mean **against the unconditional base rate of buying any bar in
  a matching downtrend context** (not just any bar), isolating the pattern's own
  information from plain "buy the dip."
- **Data snooping on chart patterns.** **Lo, Mamaysky & Wang (2000)**, *Foundations of
  Technical Analysis* (Journal of Finance), formalize testing chart patterns against a
  properly matched null; **Marshall, Young & Rose (2006)**, *Candlestick Technical
  Trading Strategies: Can They Create Value for Investors?* (Journal of Banking &
  Finance), tested the broad candlestick taxonomy on DJIA components and found no
  value beyond chance — directly on point for the candlestick zoo this study belongs
  to. **Sullivan, Timmermann & White (1999)** and **White (2000)**, *A Reality Check
  for Data Snooping* (Econometrica), motivate charging a selection/multiple-testing
  penalty before believing any single mined rule.
- **HAC inference.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  the one-sample *t* used wherever the count clears `MIN_N_FOR_TEST`; Welch (1947) for
  the decisive pattern-vs-base-rate split.

## Method lineage (the desk's shared engine)

- **Three-stars detection, two cuts.**
  [`strategy.three_stars_flags`](../three_stars_in_the_south/strategy.py),
  [`strategy.strict_three_stars_flags`](../three_stars_in_the_south/strategy.py) —
  confirmed on the close of the third star, entry at the next open (no look-ahead).
- **Downtrend-matched base rate + Welch t + label-shuffle placebo.**
  [`strategy.run_experiment`](../three_stars_in_the_south/strategy.py) — the same idiom
  as siblings 685 and 687.
- **Bonferroni correction.**
  [`strategy.bonferroni_critical`](../three_stars_in_the_south/strategy.py) for the
  four-horizon multiple-comparisons grid.
- **Deterministic synthetic control.**
  [`data.synthetic_panel`](../three_stars_in_the_south/data.py) plants forced 3-bar
  blocks (shrinking magnitude, rising lows) **only where the plain random walk is
  already in a downtrend on its own** — no artificial drift is injected anywhere else
  on the tape, so star events and the base rate are drawn from the same population —
  with a *tunable* signed post-pattern bounce (knob `edge`); with `edge = 0` the
  detector must not manufacture significance in the base-rate-relative delta across
  seeds. The offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) OHLCV for SPY + 60 long-listed US
  large-caps spanning every major sector, ~25 years each (cache-first; offline once
  cached). All headline numbers are pinned in [`docs/results.md`](results.md) and
  reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [187-three-soldiers](../../187-three-soldiers/) — **Three White Soldiers**, the
  bullish *continuation* mirror of Three Black Crows: three ascending white candles
  claimed to continue an *uptrend*. Opposite context (uptrend, not downtrend), opposite
  claim (continuation, not reversal), and no shrinking/rising-low geometry at all —
  the closest this study's title gets to a "positive" sibling, and still a different
  pattern entirely.
- [408-three-black-crows](../../408-three-black-crows/) — three consecutive falling
  candles read as a **bearish continuation** (short the crash), with *equal or growing*
  bodies, no rising-low requirement, and no reversal claim. Three stars in the south is
  often confused with it precisely because both start with "three black candles" — but
  three black crows is a **momentum** claim tested **short**, while this study is an
  **exhaustion** claim, requiring *shrinking* bodies and *rising* lows, tested **long**.
  The shapes are near opposites dressed in the same three-candle, three-black costume.
- [687-ladder-bottom](../../687-ladder-bottom/) — the desk's other black-candle-into-
  reversal bottoming pattern: **five** bars (four declining, one confirming bullish
  break), vs three stars in the south's **three** bars with no separate confirming
  candle (the third star's own shrinking/rising-low shape *is* the signal). Same
  downtrend-matched base-rate idiom, same strict/loose and `MIN_N_FOR_TEST` discipline,
  reused directly from that study.
- None of the siblings test the specific **three-shrinking-black-candles-with-rising-
  lows** shape — this study's own axis.
