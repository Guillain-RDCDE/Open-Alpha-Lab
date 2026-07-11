# References & literature map — Study 688 (Concealing Baby Swallow)

## The claim under test

- **The folklore.** "The concealing baby swallow is a **bullish reversal**: two black
  marubozu, then a black candle that gaps down and rallies back into the prior body before
  closing low, then a fourth black candle that swallows the third whole and still closes
  at a new low — this is capitulation, and the downtrend is about to end." One of the
  named formations in the Japanese candlestick canon, and by wide agreement the single
  **rarest** one in the book.
- **The canonical source.** Steve Nison, *Japanese Candlestick Charting Techniques*
  (2nd ed., 1991/2001, New York Institute of Finance / Prentice Hall) — the book that
  brought candlestick analysis to Western trading desks — names and illustrates the
  pattern. Thomas Bulkowski, *Encyclopedia of Candlestick Charts* (2008, Wiley), performs
  the closest thing to an empirical count in the popular literature and flags it
  explicitly as too infrequent to rank reliably against his other 100+ patterns.
  Investopedia and most charting-platform glossaries repeat the four-candle definition
  verbatim, none supply a real occurrence count.
- **What we test.** Whether the precise four-candle OHLC shape (loose and strict cuts)
  occurs at all, at any usable rate, across a very large, long-history US equity basket —
  and, only if it does occur often enough to say anything, whether going long the next
  open earns a positive, significant return over the next 1/5/10/20 days versus the
  unconditional base rate for "four red days in a downtrend."

## Why this is the desk's cleanest "unfalsifiable in practice" case

- **Combinatorial rarity by design.** The pattern requires **two** near-perfect
  zero-shadow bodies (marubozu) *and* a precise overlap-then-total-engulf geometry on the
  following two bars — a conjunction of five-plus independent low-probability conditions.
  Compare the desk's other candlestick teardowns: three-black-crows
  ([408](../../408-three-black-crows/)) needs three conditions, morning-star
  ([186](../../186-morning-star/)) needs a small indecision star between two bigger
  candles, ladder-bottom ([687](../../687-ladder-bottom/)) needs five candles but no
  precise overlap/engulf geometry. Concealing baby swallow stacks *both* a five-candle-
  scale combinatorial requirement *and* a precise-geometry requirement, and it shows: on
  111 tickers and ~4,957 name-years of daily bars (see `docs/results.md`), the loose cut
  fires **4** times and the strict, literature-close cut fires **0** times.
- **A pre-registered "too few to test" rule**, not a post-hoc excuse. Before running
  anything we fix :data:`strategy.MIN_N_FOR_TEST` = 8 pooled events — below that no
  *t*-statistic is computed at all (`strategy.summarize`, `strategy.welch_t`). A *t* on 4
  points, or on 0, is decoration, not evidence; the honest report is "too rare to test,"
  not a manufactured (and meaningless) significance number.
- **Why "too rare to test" earns `Signal: NONE`, not `WEAK`.** The desk's inference bar
  (`METHODOLOGY.md`) requires an autocorrelation-robust statistic clearing *t* >= 2 on the
  real tape for `REAL`, and reserves `WEAK` for a real (if fragile) point estimate.
  Here there is no testable point estimate at all — a claim that cannot be exposed to
  disconfirming evidence at any practical sample size is not weak evidence for the claim,
  it is an absence of evidence, and the desk grades that `NONE`.

## The broad evidence on candlestick patterns generally

- **Marshall, Young & Rose (2006), *Candlestick technical trading strategies: Can they
  create value for investors?* (Journal of Banking & Finance)** — the full candlestick
  taxonomy (including rarer multi-candle formations) tested on DJIA components; no value
  after accounting for data-snooping. The desk's broader candlestick teardowns
  (408, 186, 687) are consistent nulls on the *common* patterns; this study asks a
  different, prior question about a *rare* one — can it even be observed?
- **Lo, Mamaysky & Wang (2000), *Foundations of technical analysis* (Journal of
  Finance)** — kernel-smoothing evidence that *some* chart patterns carry marginal
  information, but the formations that survive are the common, well-populated ones; a
  formation this restrictive was never in scope for that kind of test.
- **Harvey, Liu & Zhu (2016), *…and the Cross-Section of Expected Returns* (Review of
  Financial Studies)** and **White (2000), *A reality check for data snooping*
  (Econometrica)** — the general case for a selection/multiple-testing penalty before
  believing any single named pattern; moot here, since the raw sample never reaches a
  size where a penalty would even apply.

## Method lineage (the desk's shared engine)

- **Precise OHLC detector, two cuts (loose + strict).**
  [`strategy.cbs_flags`](../concealing_baby_swallow/strategy.py) and
  [`strategy.strict_cbs_flags`](../concealing_baby_swallow/strategy.py) — mirrors the
  loose/strict two-cut idiom used by sibling study
  [687-ladder-bottom](../../687-ladder-bottom/).
- **Base-rate-matched long event study, no *t*-stat below the pre-registered floor.**
  [`strategy.cbs_events`](../concealing_baby_swallow/strategy.py),
  [`strategy.base_rate_events`](../concealing_baby_swallow/strategy.py),
  [`strategy.summarize`](../concealing_baby_swallow/strategy.py),
  [`strategy.welch_t`](../concealing_baby_swallow/strategy.py) (returns `None`, not a
  number, below `MIN_N_FOR_TEST`).
- **Deterministic synthetic control.**
  [`data.synthetic_panel`](../concealing_baby_swallow/data.py) plants the *exact* four-
  candle geometry at a controlled rate with a tunable post-pattern bounce; with the edge
  set to zero the detector must not manufacture significance, and it must recover a
  planted edge — proof the near-zero real-tape count is a property of the market, not a
  broken or over-strict detector.

## Data sources used here

- **yfinance** daily OHLCV (`auto_adjust=True`) for a **111-ticker** basket — SPY, QQQ,
  DIA, IWM plus 107 long-listed US large-caps spanning every major sector, cached under
  `_cache/cbs_*.parquet`. As-of **2026-06-30**; oldest bar 1962-01-02 (several names span
  the full 64.5 years). All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [408-three-black-crows](../../408-three-black-crows/) — a **three**-candle **bearish**
  pattern, tested (and found a null in the wrong direction) on a well-populated sample.
  Different direction, different cardinality, and — critically — enough events to
  actually run a test.
- [687-ladder-bottom](../../687-ladder-bottom/) — a **five**-candle **bullish** reversal
  with the same "needs a very large basket" premise, but without this study's precise
  overlap-then-engulf geometry; its own sample size (see its `docs/results.md`) sits
  below this study's, but tests the geometry, not the marubozu/gap-and-fail/engulf
  conjunction that makes the concealing baby swallow uniquely rare.
- [186-morning-star](../../186-morning-star/) — a **three**-candle bullish reversal
  (small indecision star between two larger bodies), common enough to run a full
  HAC/Bonferroni event study on 15 tickers over 16 years; found a *negative* excess, the
  opposite failure mode from this study's "too rare to even test."
- None of the siblings ask the question this study asks first: **does the pattern occur
  often enough, anywhere, to test at all?** For concealing baby swallow, on the desk's
  largest basket and longest history, the honest answer is no.
