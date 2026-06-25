# References & literature map — Study 457 (Kicker-Pattern)

## The claim under test

- **The folklore.** The **kicker** (or "kicker signal") is a two-candle reversal pattern: two
  *opposite-colour* marubozu candles separated by a **gap in the new direction**, said to ignore
  the prior trend entirely. A bullish kicker is a black (down) marubozu followed by a white (up)
  marubozu that gaps up above the prior candle's open; a bearish kicker is the mirror. The lore,
  repeated across Investopedia, StockCharts' ChartSchool and countless trading blogs, is that the
  kicker is **"one of the most reliable" reversal signals** — when it prints you trade its
  direction, no other confirmation needed.
- **The source.** Candlestick charting was introduced to the West by **Steve Nison**, *Japanese
  Candlestick Charting Techniques* (1991), which catalogues marubozu candles and gap-based
  reversals. **Thomas N. Bulkowski**, *Encyclopedia of Candlestick Charts* (2008), is the most
  cited *quantitative* catalogue: he ranks the kicker among the higher-frequency-of-reversal
  patterns in his sample — but his own statistics are unconditional (reversal frequency vs the
  pattern's direction, **not** vs a drift-matched or random baseline), which is precisely the gap
  this study fills. The marubozu primitive traces to traditional Japanese rice-trading charting
  (the "Honma" lineage popularised by Nison).
- **Variants.** Some define the kicker with a *body* gap (close-to-open) rather than an
  open-to-open gap, or relax the marubozu to "long real body"; all are affine loosenings of the
  same opposite-bodies-plus-gap geometry and inherit the same confound tested here.

## Why this is a "theory" / mechanical-proxy study

The kicker is usually drawn by eye ("is that a marubozu? is the gap big enough?"). Following the
desk's design for discretionary patterns, we encode the **tightest mechanical rule a proponent
would accept** and state the irreducible choices explicitly:

- **Objective marubozu.** body / (high − low) ≥ 0.60 — a strong candle whose wicks are at most
  ~40% of the range. (At the textbook-strict 0.80 the canonical kicker prints **only 6 times in
  21 years** across five ETFs — itself a finding; 0.60 is the loosest defensible "big body"
  threshold and yields 80 events.)
- **Objective gap.** bar *t* opens past bar *t-1*'s open in bar *t*'s own direction. No look-ahead:
  the pattern is completed on the close of bar *t* (using only *t-1* and *t*); entry is the **next
  close**.
- **The honest baseline.** Because the rule mixes longs and shorts on a drifting tape, the only
  meaningful comparison is a **direction-matched random-entry** control (same long/short mix, same
  instrument, epoch and hold). We add a **gap-scramble placebo** that keeps the candle marginal
  but randomises the gap signs — the direct test of "does the gap-in-the-new-direction matter?"

## Why a signal-vs-zero *t* is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample *t*
  of any *long* entry rule against **zero** measures that drift, not the rule. Conversely a
  *short* rule is penalised by the same drift. The desk's standing rule is *signal-vs-baseline*,
  never *signal-vs-zero* — hence the direction-matched random control. See Fama & French on the
  equity premium.
- **Data snooping on chart patterns.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis: Computational Algorithms, Statistical Inference, and Empirical Implementation*, J.
  Finance) formalise testing chart patterns against a properly matched null and find most carry
  little *conditional* information. Sullivan, Timmermann & White (1999, *Data-Snooping, Technical
  Trading Rule Performance, and the Bootstrap*, J. Finance) and White (2000, *A Reality Check for
  Data Snooping*, Econometrica) show how pattern rules manufacture significance unless raced
  against a fair benchmark with a multiple-testing correction.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the kicker-vs-random difference.

## Method lineage (the desk's shared engine)

- **Marubozu / kicker detection.** [`strategy.marubozu_flags`](../kicker_pattern/strategy.py),
  [`strategy.kicker_signals`](../kicker_pattern/strategy.py) — the mechanical geometry with the
  one-bar completion / next-close entry baked in.
- **Forward-return + HAC t + direction-matched random baseline.**
  [`strategy.forward_returns`](../kicker_pattern/strategy.py),
  [`strategy.hac_t`](../kicker_pattern/strategy.py),
  [`strategy.run_experiment`](../kicker_pattern/strategy.py).
- **Geometry placebo.** [`strategy.gap_scramble_placebo`](../kicker_pattern/strategy.py) — permute
  the open-gap signs, keep the candle marginal.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../kicker_pattern/data.py) plants
  genuine kicker formations and a real continuation (knob `edge`); with `edge = 0` the detector
  must NOT manufacture significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced
  by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../450-andrews-pitchfork`](../450-andrews-pitchfork) — the sibling chart-pattern teardown
  whose engine and random-entry idiom this study reuses verbatim.
- [`../402-engulfing-pattern`](../402-engulfing-pattern), [`../406-harami-pattern`](../406-harami-pattern)
  and the broader candlestick zoo (402–409) — the same family of two-candle reversal patterns,
  same None × Mirage destination.
- The **research-method demos** (data-mining-roulette, multiple-testing, look-ahead) frame why a
  signal-vs-zero *t* — or an unconditional reversal frequency à la Bulkowski — is not enough;
  the kicker is a clean live example of a vivid pattern with no conditional edge.
