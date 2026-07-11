# References & literature map — Study 689 (Upside Gap Two Crows)

## The claim under test

- **The folklore.** "The **upside gap two crows** is a rare but ominous bearish
  reversal: after a rally, a bullish candle is followed by a black candle that **gaps
  up** away from it — a fresh high, buyers still apparently in control — but the very
  next candle is *another* black candle that opens even higher and closes lower,
  engulfing the first crow's body. Two black 'crows' have now landed on the gap and are
  digging into it, yet **fail to fully close it**. That unresolved tension — bulls who
  gapped the stock higher, bears now clawing it back down but not all the way — is read
  as exhaustion: **the top is in**. Sell, or short." One of the rarer three-candle
  patterns in the canon, prized by chartists precisely because it is uncommon and
  "looks" dramatic on a chart.
- **The canonical source.** Steve Nison, *Japanese Candlestick Charting Techniques*
  (1991, New York Institute of Finance) codifies the upside gap two crows among the
  bearish reversal figures; Thomas Bulkowski's *Encyclopedia of Candlestick Charts*
  (2008, Wiley) catalogs it with historical hit-rate statistics from his own screens and
  flags it as one of the **least reliable** named patterns in his own taxonomy (he rates
  it near the bottom of his performance rankings) — a useful "the literature itself is
  skeptical" starting point that this study measures independently, on its own basket,
  protocol and horizon set.
- **What we test.** Whether shorting the precise OHLC pattern, entered one day after
  the confirming close, earns a positive (and significant) return over the next
  1/5/10 days — i.e. whether the "top" actually holds — measured *fairly* against what
  the same basket does on an unconditional day (not against zero; see the note in
  [`docs/results.md`](results.md) on why a plain *t*-vs-zero over-states the case).

## Why the bearish reading is mechanically suspect

- **Selection on a fall.** The confirming candle is, by construction, the second of two
  consecutive black (down) bodies — a stock that has just dropped for two sessions in a
  row after an initial gap-up spike. Short-horizon **mean reversion** then tends to work
  *against* a fresh short, exactly the mechanism the desk's sibling studies on
  [408-three-black-crows](../../408-three-black-crows/) and
  [683-evening-star](../../683-evening-star/) document for their own multi-black-candle
  bearish patterns. Jegadeesh (1990, *Evidence of predictable behavior of security
  returns*, JF) and Lehmann (1990, *Fads, martingales, and market efficiency*, QJE)
  document short-term reversal in individual stocks — the headwind a naive post-pattern
  short faces here too.
- **The unconditional drift.** Equities drift **up** on average, so any *short* starts
  with a base-rate headwind; a signed-short return that is negative but statistically
  indistinguishable from the basket's own unconditional short-of-everything mean is
  **not** evidence of a bearish edge — it is the market's ordinary drift. This is why
  the study's certifying statistic is a **Welch *t* against the unconditional base**,
  not a one-sample *t* against zero (see [`strategy.py`](../upside_gap_two_crows/strategy.py)
  and the worked example in the synthetic control, where a **known** zero-edge panel
  still shows the vs-zero HAC *t* moving around from its own embedded drift, not the
  pattern).

## The broad evidence on candlestick patterns

- **Marshall, Young & Rose (2006), *Candlestick technical trading strategies: Can they
  create value for investors?* (Journal of Banking & Finance)** — test the full
  candlestick taxonomy on DJIA components and find **no value** after accounting for
  data-snooping. Our large-cap null is consistent with this.
- **Horton (2009)** and **Marshall, Young & Cahan (2008)** extend the null to other
  markets and other rare reversal figures.
- **Lo, Mamaysky & Wang (2000), *Foundations of technical analysis* (JF)** — a careful
  kernel-smoothing study that finds *some* chart patterns carry marginal information;
  the rarer, multi-candle "two crows" family of figures is not among the clean
  survivors once costs and multiple testing are charged.
- **Bulkowski's own screens** (thepatternsite.com / *Encyclopedia of Candlestick
  Charts*) rank the upside gap two crows near the bottom of his reliability tables even
  by the standards of a book that is broadly sympathetic to candlestick analysis — this
  study's independent null is not an outlier finding.

## Why a high event count still needs a placebo + HAC + a Bonferroni correction

- **Newey-West (1987) HAC** standard errors for the signed-short mean (event ordering
  can carry serial dependence within a name; see
  [`strategy.hac_t`](../upside_gap_two_crows/strategy.py)). Reported as an
  *informational* cross-check — the decisive number is the **Welch *t*** against the
  basket's own unconditional pool, since a plain HAC-vs-zero test is contaminated by the
  tape's ordinary up-drift (see above).
- **Coin-flip label-shuffle placebo** (Fisher's randomization logic; Efron & Tibshirani,
  *An Introduction to the Bootstrap*, 1993) — draw the same event count from the
  unconditional pool, sign each by a fair coin, and ask how often a random pick beats
  the observed mean. See
  [`strategy.placebo_pvalue`](../upside_gap_two_crows/strategy.py).
- **Bonferroni across the basket.** With 26 tickers (of 30) carrying ≥ 6 events tested
  independently for a per-name version of the same effect, the two-sided significance
  bar must widen to `|t| >= z(1 - 0.025/26) ~= 3.10` (see
  [`strategy.bonferroni_z`](../upside_gap_two_crows/strategy.py) and
  [`strategy.per_ticker_stats`](../upside_gap_two_crows/strategy.py)) — otherwise the
  single loudest name in the basket gets mistaken for "the" signal. Harvey, Liu & Zhu
  (2016, *…and the Cross-Section of Expected Returns*, RFS) and White (2000, *A reality
  check for data snooping*, Econometrica) motivate the general principle.

## Method lineage (the desk's shared engine)

- **Precise OHLC detector + signed event study.**
  [`strategy.is_upside_gap_two_crows`](../upside_gap_two_crows/strategy.py) and
  [`strategy.collect_events`](../upside_gap_two_crows/strategy.py) — a bullish body,
  then a black body gapping up from it, then a second black body that opens above the
  first crow's open and closes below the first crow's close (engulfing it from above)
  while staying above the first candle's close (the gap is never fully filled), signed
  short, with one execution lag.
- **Drift-neutral inference.** [`strategy.welch_t`](../upside_gap_two_crows/strategy.py)
  (the decisive comparison), [`strategy.hac_t`](../upside_gap_two_crows/strategy.py)
  (informational, vs zero), coin-flip placebo, and the Bonferroni bar — the same
  engine lineage as [683-evening-star](../../683-evening-star/) and
  [408-three-black-crows](../../408-three-black-crows/).
- **Deterministic synthetic control.**
  [`data.synthetic_panel`](../upside_gap_two_crows/data.py) plants a known post-pattern
  crash on a panel with its own embedded up-drift and overnight-gap noise (mirroring the
  real basket's own contamination risk and the pattern's need for a genuine gap); with
  the edge set to zero the drift-neutral inference must NOT manufacture significance
  (checked over 20 seeds) — the offline core runs with no network.

## Data sources used here

- **yfinance** daily OHLCV (`auto_adjust=True`) for a fixed 30-name liquid large-cap +
  SPY basket, 2005-01-03 → 2026-06-30, cached under `_cache/ugtc_*.parquet`. All
  headline numbers are pinned in [`docs/results.md`](results.md) (fingerprints
  `b1038f477d11` / `67c54c8a3aa2`) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- **[408-three-black-crows](../../408-three-black-crows/)** — **three consecutive** tall
  red candles with **no gap requirement at all**. Shares the "crow" naming and the
  post-fall mean-reversion headwind this study documents too, but a structurally
  different figure: no bullish first candle, no gap, three bodies instead of a
  bullish-then-two-black triple.
- **[683-evening-star](../../683-evening-star/)** — a tall bullish candle, a **small
  star** (any color) that gaps up, then a tall bearish candle closing deep into the
  first candle's body. The star's *body size* (small) is the defining feature, not a
  gap that survives two subsequent down candles; this study's second and third candles
  are both full-sized **black** bodies, and the gap must specifically survive **two**
  bearish candles without closing — a different, stricter geometry.
- **[417-island-reversal](../../417-island-reversal/)** — a **two-gap** figure (an
  exhaustion gap, a cluster of stranded bars, then a *second, opposite-direction* gap
  that seals the island) on a much longer, name-agnostic detection window (1–3 island
  bars, 5–40 day horizons). This study has exactly **one** gap (the upside gap) that the
  two crows try, and fail, to close — there is no second, opposite gap sealing anything.
- **[407-dark-cloud-piercing](../../407-dark-cloud-piercing/)** — a **two-candle** pair
  (Dark Cloud Cover / Piercing Line): a long body, then a candle that gaps the *other*
  way and closes back past the **midpoint** (not past the close) of the first body. Two
  candles, not three, a midpoint-penetration rule rather than an engulf-but-don't-fill
  rule, and — critically — the gap in Dark Cloud Cover/Piercing is **against** the prior
  trend (a reversal gap), the opposite geometry of the upside gap two crows' gap **with**
  the trend that the two crows then fail to close.

None of the siblings run the dedicated three-candle, gap-then-two-crows-fail-to-close
detector against an unconditional-base, Bonferroni-corrected, cost-and-borrow-charged
bar — this study's own axis.
