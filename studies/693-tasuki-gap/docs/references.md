# References & literature map — Study 693 (Tasuki Gap)

## The claim under test

- **The folklore.** The **tasuki gap** (upside and downside) is one of the older named
  continuation figures in the Japanese candlestick canon: a trending market gaps in the
  trend's direction on a same-color candle, then a small candle of the *opposite* color
  opens inside that gapping candle's body and pulls back — but crucially **fails to
  close the gap**. The reading: if the market can't even close a small, fresh gap
  against the trend, the trend still has force left, and it should **resume** once the
  pullback is done. Bullish version = upside tasuki gap (go long); bearish mirror =
  downside tasuki gap (go short).
- **The canonical source.** Steve Nison, *Japanese Candlestick Charting Techniques*
  (1991/2001, New York Institute of Finance) codifies both the upside and downside
  tasuki gap among the continuation figures. Thomas Bulkowski's *Encyclopedia of
  Candlestick Charts* (2008, Wiley) catalogs the pattern with his own historical
  hit-rate screens; this study measures it independently, on its own basket, protocol
  and horizon set, and does not borrow his numbers.
- **What we test.** Whether entering in the predicted trend direction, one day after
  the confirming close, earns a positive (and significant) return over the next
  1/5/10/20 days — i.e. whether the trend actually *does* resume — measured *fairly*
  against what the same basket does on an unconditional day in the same direction (not
  against zero; see the note in [`docs/results.md`](results.md) on why a plain
  *t*-vs-zero over- or under-states the case depending on direction).

## Why the continuation reading is mechanically suspect

- **A gap plus a same-day-or-two pullback is exactly the setup short-horizon reversal
  research warns about.** Jegadeesh (1990, *Evidence of predictable behavior of
  security returns*, JF) and Lehmann (1990, *Fads, martingales, and market efficiency*,
  QJE) document short-term reversal in individual stocks — a headwind that would work
  *against* a naive continuation trade exactly on the horizons (1–10 days) tasuki gaps
  are supposed to resolve on. This study's own down-leg result (a 5/10-day dip on the
  downside tasuki, i.e. a *reversal* not a continuation) is consistent with that
  literature, not a coincidence.
- **The unconditional drift cuts both ways.** Equities drift **up** on average, so a
  *long* continuation event needs to beat more than "the market went up anyway," and a
  *short* continuation event needs to beat more than "shorting anything fights the
  market's up-drift." This is why the study's certifying statistic is a **Welch *t*
  against the direction-matched unconditional base** (long events vs the plain pool,
  short events vs its negation), not a one-sample *t* against zero (see
  [`strategy.py`](../tasuki_gap/strategy.py) and the worked example in the synthetic
  control, where a **known** zero-edge panel still moves the vs-zero HAC *t* around from
  its own embedded drift, not the pattern).

## The broad evidence on candlestick patterns

- **Marshall, Young & Rose (2006), *Candlestick technical trading strategies: Can they
  create value for investors?* (Journal of Banking & Finance)** — test the full
  candlestick taxonomy on DJIA components and find **no value** after accounting for
  data-snooping. Our large-cap null is consistent with this.
- **Horton (2009)** and **Marshall, Young & Cahan (2008)** extend the null to other
  markets and other named continuation/reversal figures.
- **Lo, Mamaysky & Wang (2000), *Foundations of technical analysis* (JF)** — a careful
  kernel-smoothing study that finds *some* chart patterns carry marginal information;
  gap-continuation figures specifically are not among the clean survivors once costs and
  multiple testing are charged.

## Why a high event count still needs a placebo + HAC + a Bonferroni correction

- **Newey-West (1987) HAC** standard errors for the trend-signed mean (event ordering
  can carry serial dependence within a name; see
  [`strategy.hac_t`](../tasuki_gap/strategy.py)). Reported as an *informational*
  cross-check — the decisive number is the **Welch *t*** against the direction-matched
  unconditional pool, since a plain HAC-vs-zero test is contaminated by the tape's
  ordinary drift and by which direction (long/short) the event happens to be (see
  above).
- **Mix-matched coin-flip placebo** (Fisher's randomization logic; Efron & Tibshirani,
  *An Introduction to the Bootstrap*, 1993) — draw the same event count from the
  unconditional pool, sign each by a coin weighted to the observed up/down mix (not a
  plain 50/50 coin, since the real sample is a specific long/short blend), and ask how
  often a random pick beats the observed mean. See
  [`strategy.placebo_pvalue`](../tasuki_gap/strategy.py).
- **Bonferroni across the basket.** With 30 tickers carrying ≥ 6 events tested
  independently for a per-name version of the same effect, the two-sided significance
  bar must widen to `|t| >= z(1 - 0.025/30) ~= 3.14` (see
  [`strategy.bonferroni_z`](../tasuki_gap/strategy.py) and
  [`strategy.per_ticker_stats`](../tasuki_gap/strategy.py)) — otherwise the single
  loudest name in the basket gets mistaken for "the" signal. Harvey, Liu & Zhu (2016,
  *…and the Cross-Section of Expected Returns*, RFS) and White (2000, *A reality check
  for data snooping*, Econometrica) motivate the general principle.

## Method lineage (the desk's shared engine)

- **Precise OHLC detector + trend-signed event study.**
  [`strategy.tasuki_signal`](../tasuki_gap/strategy.py) and
  [`strategy.collect_events`](../tasuki_gap/strategy.py) — two same-color candles
  gapping in the trend direction, then an opposite-color candle that opens inside the
  second candle's body and closes back inside the gap without filling it, signed toward
  the predicted trend direction, with one execution lag.
- **Drift-neutral, direction-matched inference.**
  [`strategy.welch_t`](../tasuki_gap/strategy.py) (the decisive comparison, matched to
  each event's own long/short direction), [`strategy.hac_t`](../tasuki_gap/strategy.py)
  (informational, vs zero), the mix-matched coin placebo, and the Bonferroni bar — the
  same engine lineage as [689-upside-gap-two-crows](../../689-upside-gap-two-crows/),
  [683-evening-star](../../683-evening-star/) and
  [408-three-black-crows](../../408-three-black-crows/).
- **Deterministic synthetic control.**
  [`data.synthetic_panel`](../tasuki_gap/data.py) plants a known post-pattern
  continuation on a panel with its own embedded drift and overnight-gap noise
  (mirroring the real basket's own contamination risk and the pattern's need for a
  genuine gap); with the edge set to zero the drift-neutral inference must NOT
  manufacture significance (checked over 20 seeds) — the offline core runs with no
  network.

## Data sources used here

- **yfinance** daily OHLCV (`auto_adjust=True`) for a fixed 30-name liquid large-cap +
  SPY basket, 2005-01-03 → 2026-06-30, cached under `_cache/tg_*.parquet`. All headline
  numbers are pinned in [`docs/results.md`](results.md) (fingerprints `8af479f8c79c` /
  `aa438a4d53ea`) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- **[74-mind-the-gap](../../74-mind-the-gap/)** — the general opening-gap **fill-rate**
  question (does a gap fill by session end, at all, regardless of shape or context)
  across all gap sizes. This study is the opposite prediction on a *specific* three-bar
  shape: the tasuki gap's whole thesis is that the gap does **not** fill and the trend
  **continues** — it is a directional continuation claim conditioned on a precise
  two-same-color-then-opposite-pullback geometry, not a base-rate fill-frequency study.
- **[455-three-methods](../../455-three-methods/)** — rising/falling **three methods**:
  a long anchor candle, **three** small candles that stay *inside* the anchor's
  high–low range (no gap required or even allowed), then a confirming breakout candle.
  Also a continuation claim, but the geometry has no gap at all and one more candle in
  the consolidation — a structurally different pattern this study does not test.
- **[417-island-reversal](../../417-island-reversal/)** — a **two-gap** figure (an
  exhaustion gap, a cluster of stranded bars, then a *second, opposite-direction* gap
  sealing the island) predicting a **reversal**, the opposite claim to tasuki's
  continuation. This study has exactly **one** gap that the pullback candle tries, and
  fails, to close — there is no second, opposite gap, and the predicted outcome is the
  trend resuming, not reversing.

None of the siblings run the dedicated three-candle, gap-then-incomplete-pullback,
trend-**continuation** detector against a direction-matched unconditional base,
Bonferroni-corrected, cost-and-borrow-charged bar — this study's own axis.
