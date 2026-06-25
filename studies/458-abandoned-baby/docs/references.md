# References & literature map — Study 458 (Abandoned-Baby / island-doji reversal)

## The claim under test

- **The folklore.** The *abandoned baby* is a three-candle **island reversal**. In the bullish
  version (a bottom): a down candle, then a **doji** (open ≈ close) that **gaps down** away from
  it, then a candle that **gaps up** away from the doji and closes higher. The doji is the
  "abandoned baby" — marooned on its own price island by gaps on *both* sides. Lore says the
  island doji *calls the turn*, so you **buy the up-gap confirmation** and ride the reversal. It
  is widely billed as one of the **rarest and most reliable** candlestick patterns.
- **The source.** **Steve Nison** introduced Japanese candlestick patterns to the West in
  *Japanese Candlestick Charting Techniques* (1991) and *Beyond Candlesticks* (1994), translating
  the centuries-old Japanese rice-trading tradition (often traced to **Munehisa Homma**, 18th c.).
  The abandoned baby is Nison's name for the *sute go* / island-doji reversal — the candlestick
  analogue of the classic Western **island reversal** (an isolated price island left by two gaps).
  **Thomas Bulkowski** (*Encyclopedia of Candlestick Charts*, 2008) tabulates its (very small-
  sample) historical performance and ranks it highly on "reliability".
- **Variants.** The *bearish* abandoned baby is the mirror (an up-trend, an up-gapped doji, a
  down-gapped confirmation). The *evening/morning doji star* is the same three-bar geometry
  **without** the strict requirement of gaps on both sides — i.e. the abandoned baby is the
  gapped, "island" special case. All share the doji-after-a-move core tested here.

## Why this is a "theory" / mechanical-proxy study

The abandoned baby is *semi-subjective*: a chartist eyeballs "a doji", "a gap", "a trend".
Following the desk's design for this kind, we encode the **tightest mechanical rule a proponent
would accept** and state the irreducible choices explicitly:

- **Objective doji.** Body ≤ 10% of the bar's high-low range — the standard mechanical doji test.
- **Objective trend context.** The bullish version requires a *prior decline*, so bar A must be a
  down candle closing **below its SMA-20**; otherwise the "reversal" has nothing to reverse.
- **Objective island gaps.** Both-sided gaps. Daily ETFs almost never produce the strict
  *full-range* island (a few entries in 21 years), so the charitable encoding gaps the **bodies**
  (open/close) on both sides — precisely the island a candlestick reader sees. Even so the
  pattern fires only **80** times across five tapes: it is genuinely rare, and a thin sample is
  itself a caveat on every statistic.
- **The honest baseline.** The only meaningful comparison on an upward-drifting index is the
  **random-entry** control (same instrument, epoch, hold), because *any* long entry inherits the
  drift. We add a **gap-scramble placebo** that keeps the doji-after-a-decline candidate pool but
  destroys the two-sided-gap geometry — the direct test of "does the *island* matter?"

## Why the high one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample *t*
  of a long-only entry rule against **zero** measures that drift, not the rule. See Fama & French
  on the equity premium; the desk's standing rule is *signal-vs-baseline*, never *signal-vs-zero*.
  The 60-day one-sample *t* of +3.54 here is mostly the longest, most drift-loaded hold.
- **Data snooping on chart patterns.** **Lo, Mamaysky & Wang (2000)**, *Foundations of Technical
  Analysis* (Journal of Finance), formalize testing chart patterns against a properly matched
  null and find most "patterns" add little once the benchmark is fair. **Sullivan, Timmermann &
  White (1999)**, *Data-Snooping, Technical Trading Rule Performance, and the Bootstrap* (JF), and
  **White (2000)**, *A Reality Check for Data Snooping* (Econometrica), show how rules mined from
  past price manufacture significance unless raced against a fair benchmark — which, here, is the
  random-entry baseline and the gap-scramble placebo.
- **Small samples.** With ~16 trades per instrument, any single-horizon "hit" (the 60-day Welch
  *t* = 2.58) is exactly where a multiple-testing fluke would surface; the desk's
  research-method demos (multiple-testing, data-mining-roulette) frame why one horizon out of
  four is not a discovery.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the pattern-vs-random difference.

## Method lineage (the desk's shared engine)

- **Pattern detection.** [`strategy.abandoned_baby_entries`](../abandoned_baby/strategy.py) —
  the mechanical doji + down-trend + two-sided-gap test, confirmed on bar C with a next-close
  entry (no look-ahead).
- **Forward-return + HAC t + random baseline.** [`strategy.forward_returns`](../abandoned_baby/strategy.py),
  [`strategy.hac_t`](../abandoned_baby/strategy.py), [`strategy.run_experiment`](../abandoned_baby/strategy.py).
- **Geometry placebo.** [`strategy.gap_scramble_placebo`](../abandoned_baby/strategy.py) —
  keep the doji-after-a-decline pool, destroy the island gaps, draw entries at random.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../abandoned_baby/data.py)
  plants real island bottoms (knob `edge`); with `edge = 0` the detector must NOT manufacture
  significance (it sits at *t* = −0.19) — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced
  by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../../450-andrews-pitchfork`](../../450-andrews-pitchfork) — the same chart-tool teardown
  idiom (mechanical encoding, random-entry baseline, geometry placebo, synthetic control).
- [`../../405-doji-reversal`](../../405-doji-reversal),
  [`../../402-engulfing-pattern`](../../402-engulfing-pattern) and the broader candlestick zoo
  (402–409) — most land None/Weak × Mirage/Fragile for the same reason: a two/three-bar shape
  fitted to past price re-describes the drift.
- The **research-method demos** (multiple-testing, data-mining-roulette, look-ahead) frame why a
  signal-vs-zero *t* — or a single significant horizon out of four — is not evidence; the
  abandoned baby is a clean live example of a thin-sample, drift-loaded near-miss.
