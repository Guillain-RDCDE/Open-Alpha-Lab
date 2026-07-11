# References & literature map — Study 685 (Tri-Star Doji)

## The claim under test

- **The folklore.** The **tri-star** is candlestick lore's rarest reversal claim: **three
  dojis in a row**. **Steve Nison**, *Japanese Candlestick Charting Techniques* (1991, 2nd ed.
  2001) — the book that introduced the Japanese rice-trading candlestick tradition to the
  West — describes it as an extremely rare pattern in which three consecutive sessions of
  pure indecision (open ≈ close) mark not a minor pause but a *major* trend exhaustion; the
  purist reading requires the middle doji to **gap away** from both its neighbours (a true
  "island star"), the shape that makes it legendarily uncommon. **Thomas Bulkowski**
  (*Encyclopedia of Candlestick Charts*, 2008) tabulates it among the patterns too rare to
  rank reliably on his own multi-decade sample — the honest starting point for this study.
- **The steelman.** *A tri-star, conditioned on the trend into it, predicts a forward
  reversal that beats the unconditional against-the-move base rate, net of costs — and,
  because it is billed as "major", the effect should show up over multi-week horizons, not
  just the next session.*

## Why this is a mechanical-proxy study, and the two-cut design

The tri-star is *semi-subjective*: a chartist eyeballs "three dojis" and "a gap". Following
the desk's design for this kind of claim (see the sibling studies below), we encode the
**tightest mechanical rule a proponent would accept** and report **two cuts** side by side,
stated as decisions, not buried as details:

- **Strict / literature-faithful.** Doji = body ≤ 10% of the bar's high-low range (the
  standard mechanical cut); tri-star = three consecutive dojis whose **middle bar's range
  clears both neighbours** (a true gapped island star, Nison's purist reading). This is the
  primary claim under test — and it is genuinely as rare as the lore says.
- **Loose.** Three consecutive dojis with **no gap requirement** — the plain reading a
  looser proponent might accept, common enough to actually run inference on. Reported as the
  honest power/robustness check: even granting the loose reading, is there anything here?

## Why the honest sample-size rule matters here specifically

- **A rare pattern needs many names, many years — and even then may not be enough.**
  Pooled across a 60-name, ~25-year panel (yfinance daily), the loose cut fires hundreds of
  times, but the strict, gap-faithful tri-star fires only a **handful** of times. Reporting a
  *t*-statistic on a handful of trades is decoration, not evidence — the desk's rule (see
  `strategy.MIN_N_FOR_TEST`) is to **not compute one** below a minimum count and say plainly
  "too few to test," rather than dress up noise as a verdict either way.
- **Multiple horizons, multiple looks.** Four forward horizons (5/10/20/60 days) means four
  simultaneous hypotheses; a naïve |t| ≥ 2 lets roughly one in five of them clear by pure
  chance. We apply a **Bonferroni correction** (k = 4) and report which, if any, horizon
  survives it — the same discipline sibling study 186 uses for its two-pattern × two-horizon
  grid.

## Why a high one-sample *t* would not be evidence even if it appeared

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample
  *t* of a directional entry rule against **zero** measures that drift, not the pattern. The
  desk's standing rule is *signal-vs-baseline*, never *signal-vs-zero* — the decisive number
  here is the tri-star reversal mean **against the unconditional base rate** (the same
  against-the-trend bet on every eligible bar), exactly as sibling study 405 established.
- **Data snooping on chart patterns.** **Lo, Mamaysky & Wang (2000)**, *Foundations of
  Technical Analysis* (Journal of Finance), formalize testing chart patterns against a
  properly matched null; **Sullivan, Timmermann & White (1999)**, *Data-Snooping, Technical
  Trading Rule Performance, and the Bootstrap* (Journal of Finance), and **White (2000)**,
  *A Reality Check for Data Snooping* (Econometrica), show how rules mined from past price
  manufacture significance unless raced against a fair benchmark. **Marshall, Young & Rose
  (2006)**, *Candlestick Technical Trading Strategies: Can They Create Value for Investors?*
  (Journal of Banking & Finance), tested candlestick patterns on the DJIA components and
  found no value beyond chance — directly on point for the whole candlestick zoo this study
  belongs to.
- **HAC inference.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) — the
  one-sample *t* used wherever the count clears :data:`MIN_N_FOR_TEST`.

## Method lineage (the desk's shared engine)

- **Doji / tri-star detection.** [`strategy.doji_flags`](../tri_star_doji/strategy.py),
  [`strategy.tri_star_flags`](../tri_star_doji/strategy.py),
  [`strategy.gapped_star_flags`](../tri_star_doji/strategy.py) — the loose and strict cuts,
  confirmed on the close of the third doji, entry at the next open (no look-ahead).
- **Base-rate comparison + HAC t + label-shuffle placebo.**
  [`strategy.run_experiment`](../tri_star_doji/strategy.py) — the same idiom as sibling 405.
- **Bonferroni correction.** [`strategy.bonferroni_critical`](../tri_star_doji/strategy.py)
  for the four-horizon multiple-comparisons grid, the idiom sibling 186 uses.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../tri_star_doji/data.py)
  plants forced 3-bar doji blocks with a *tunable* signed reversal (knob `edge`); with
  `edge = 0` the detector must not manufacture significance in the base-rate-relative delta
  across seeds — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) OHLCV for SPY + 60 long-listed US large-caps
  spanning every major sector, ~25 years each (cache-first; offline once cached). All
  headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [405-doji-reversal](../../405-doji-reversal/) — the **single-doji** reversal claim, on the
  same basket idiom (base-rate comparison, label-shuffle placebo). This study is its
  three-in-a-row escalation: does *stacking* the indecision candle make the claim any more
  real? (Short answer, spoiler-free here: no more than one doji did.)
- [186-morning-star](../../186-morning-star/) — a **different** three-candle reversal
  (large-small-large, directional bodies, not three dojis) tested against a random-day
  baseline. Shares the three-bar idiom and the Bonferroni discipline, not the pattern.
- [458-abandoned-baby](../../458-abandoned-baby/) — the **doji-gap** reversal: *one* doji
  island-gapped on both sides between a down candle and an up candle. The tri-star's closest
  cousin — both require a gapped doji "island" — but the abandoned baby's gaps are against
  *directional* candles either side, while the tri-star's are against *two more dojis*. This
  study reuses the abandoned baby's "strict gap vs loose no-gap" honesty split and its
  MIN_N_FOR_TEST discipline directly.
- None of the siblings test **three consecutive dojis** — the escalation this study's own
  axis is built on.
