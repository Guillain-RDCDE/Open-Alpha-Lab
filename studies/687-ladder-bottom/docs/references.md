# References & literature map — Study 687 (Ladder Bottom)

## The claim under test

- **The folklore.** The **ladder bottom** is a five-candle bullish reversal from the
  Japanese candlestick canon: **Steve Nison**, *Japanese Candlestick Charting Techniques*
  (1991, 2nd ed. 2001) — describes a downtrend producing four consecutive falling
  (bearish) candles — a visible "ladder" stepping down, the fourth rung showing a
  telltale long upper shadow as buyers start to push back — and then a fifth, bullish
  candle that breaks the decline. The claim: this specific five-bar shape marks the
  bottom, not just any run of red days followed by a green one. **Thomas Bulkowski**
  (*Encyclopedia of Candlestick Charts*, 2008) lists it among the rarer bottoming
  patterns and cautions that its historical sample sizes are thin even on his own
  multi-decade tape.
- **The steelman.** *A ladder bottom, entered the session after it confirms, earns a
  forward long return that beats the unconditional base rate of buying any bar that is
  already in a matching downtrend — net of costs — and, because the claim is that the
  downtrend specifically **ends**, the effect should persist over more than a single
  session.*

## Why this is a mechanical-proxy study, and the two-cut design

Like its rarer candlestick cousins, the ladder bottom is *semi-subjective*: a chartist
eyeballs "four falling candles" and "a reversal." Following the desk's design for this
kind of claim (see the sibling studies below), we encode the tightest mechanical rule a
proponent would accept and report **two cuts** side by side:

- **Loose.** Four consecutive bearish candles with strictly descending closes, sitting
  in a genuine prior downtrend, followed by a bullish candle that closes above the
  fourth rung's close. Common enough to run inference on (n in the low thousands, pooled).
- **Strict, literature-closer.** The loose cut, plus (a) the first three rungs close
  near their lows (small lower shadows — committed selling, not indecision), (b) the
  fourth rung shows a **longer upper shadow** than the third (the textbook "warning
  wick" Nison describes — buyers pushing back just before the reversal), and (c) the
  fifth candle **gaps up** at the open. This is the primary claim under test, and — as
  with the desk's other rare candlestick patterns — it is genuinely uncommon (about once
  per 19 ticker-years pooled).

## Why the honest sample-size rule and Bonferroni correction matter here

- **Four horizons, multiple looks.** We read forward 1/5/10/20-day returns — four
  simultaneous hypotheses. At the usual *α* = 5% level, roughly one spurious hit is
  expected every 20 independent tries; we apply a **Bonferroni correction** (k = 4,
  critical |*t*| ≥ **2.50**) and report which, if any, horizon survives it — the same
  discipline sibling studies 186 (morning-/evening-star) and 685 (tri-star doji) use for
  their own multi-horizon grids.
- **Below `MIN_N_FOR_TEST` (8 pooled events), no *t*-stat is computed at all** — a *t* on
  a handful of trades is decoration, not evidence. The strict cut here clears that bar
  (n = 81) but is still small enough that a single outlier event can move the mean a
  great deal; both cuts' full event lists are reproducible from
  [`examples/verify.py`](../examples/verify.py).

## Why a high one-sample *t* would not be evidence even if it appeared

- **Drift / beta and downtrend mean reversion.** US equities carry positive
  unconditional drift, and a bar that is already in a downtrend has its own mean-
  reversion tendency independent of any five-candle shape. The desk's standing rule is
  *signal-vs-baseline*, never *signal-vs-zero* — the decisive number here is the ladder-
  bottom reversal mean **against the unconditional base rate of buying any bar in a
  matching downtrend context** (not just any bar), isolating the pattern's own
  information from plain "buy the dip."
- **Data snooping on chart patterns.** **Lo, Mamaysky & Wang (2000)**, *Foundations of
  Technical Analysis* (Journal of Finance), formalize testing chart patterns against a
  properly matched null; **Marshall, Young & Rose (2006)**, *Candlestick Technical
  Trading Strategies: Can They Create Value for Investors?* (Journal of Banking &
  Finance), tested the broad candlestick taxonomy on DJIA components and found no value
  beyond chance — directly on point for the candlestick zoo this study belongs to.
  **Sullivan, Timmermann & White (1999)** and **White (2000)**, *A Reality Check for
  Data Snooping* (Econometrica), motivate charging a selection/multiple-testing penalty
  before believing any single mined rule.
- **HAC inference.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  the one-sample *t* used wherever the count clears `MIN_N_FOR_TEST`; Welch (1947) for
  the decisive pattern-vs-base-rate split.

## Method lineage (the desk's shared engine)

- **Ladder-bottom detection, two cuts.**
  [`strategy.ladder_bottom_flags`](../ladder_bottom/strategy.py),
  [`strategy.strict_ladder_flags`](../ladder_bottom/strategy.py) — confirmed on the
  close of the fifth candle, entry at the next open (no look-ahead).
- **Downtrend-matched base rate + Welch t + label-shuffle placebo.**
  [`strategy.run_experiment`](../ladder_bottom/strategy.py) — the same idiom as sibling
  685.
- **Bonferroni correction.**
  [`strategy.bonferroni_critical`](../ladder_bottom/strategy.py) for the four-horizon
  multiple-comparisons grid, the idiom siblings 186 and 685 use.
- **Deterministic synthetic control.**
  [`data.synthetic_panel`](../ladder_bottom/data.py) plants forced 5-bar ladder blocks
  (an engineered downtrend, four declining candles, one reversal candle) with a
  *tunable* signed post-pattern bounce (knob `edge`); with `edge = 0` the detector must
  not manufacture significance in the base-rate-relative delta across seeds — the
  offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) OHLCV for SPY + 60 long-listed US
  large-caps spanning every major sector, ~25 years each (cache-first; offline once
  cached). All headline numbers are pinned in [`docs/results.md`](results.md) and
  reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [455-three-methods](../../455-three-methods/) — **Rising/Falling Three Methods**, the
  desk's other five-candle pattern. Superficially similar candle *count*, but a
  **continuation** claim (a pause inside a trend, then the trend resumes) tested on 5
  liquid index ETFs — not a **reversal** claim on individual large-caps ending a
  downtrend. Different shape (anchor + three small inside candles + confirm), different
  direction claim, different universe.
- [408-three-black-crows](../../408-three-black-crows/) — three consecutive falling
  candles read as a **bearish continuation** (short the crash). This study's four
  falling candles are the *setup*, not the *signal* — the ladder bottom's claim is about
  the fifth, reversing candle, and it trades **long**, the opposite side from 408.
- [186-morning-star](../../186-morning-star/) — a **different** three-candle bullish
  reversal (large-bearish, small-indecision star, large-bullish) tested against a
  random-day baseline. Shares the "reversal after a downtrend, random/base-rate
  baseline, Bonferroni" idiom, not the five-candle shape.
- [685-tri-star-doji](../../685-tri-star-doji/) — three consecutive **dojis**
  (indecision candles) as a bidirectional major-reversal claim, with the same
  strict-vs-loose / `MIN_N_FOR_TEST` / Bonferroni discipline this study reuses directly.
  Different candles (dojis vs directional bodies), different candle count (3 vs 5), and
  ladder bottom's claim is one-directional (long only), not sign-matched to the prior
  trend.
- None of the siblings test the specific **four-declining-then-one-reversing** five-bar
  shape — this study's own axis.
