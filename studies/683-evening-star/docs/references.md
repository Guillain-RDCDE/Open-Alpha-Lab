# References & literature map — Study 683 (Evening-Star)

## The claim under test

- **The folklore.** "The **evening star** is a bearish reversal: after an uptrend, a tall
  bullish candle, then a small-bodied 'star' that gaps **up** — the buyers running out of
  conviction — then a tall bearish candle closing deep back into the first candle's body,
  mark **the top**. Sell or short." Together with its bullish mirror (the morning star), it
  is among the most cited "big three" candlestick reversal patterns.
- **The canonical source.** Steve Nison, *Japanese Candlestick Charting Techniques* (1991,
  New York Institute of Finance) — the book that introduced candlestick analysis to Western
  traders — codifies the evening star (and the morning star) as a top/bottom reversal
  signal, with the "closes at least halfway into the first candle's body" penetration rule
  used here. Repeated verbatim by Investopedia ("Evening Star Candlestick Pattern"),
  StockCharts, and most charting platforms.
- **What we test.** Whether shorting the precise OHLC pattern, entered one day after the
  confirming close, earns a positive (and significant) return over the next 1/5/10 days —
  i.e. whether the "top" actually holds — measured *fairly* against what the same basket
  does on an unconditional day (not against zero; see the note in
  [`docs/results.md`](results.md) on why a plain *t*-vs-zero over-states the case).

## Why the bearish reading is mechanically suspect

- **Selection on a fall.** The confirming candle is, by construction, a big **down** day —
  three consecutive candles that pre-select a stock that has just dropped sharply. Short-
  horizon **mean reversion** then tends to work *against* a fresh short, exactly the
  mechanism the desk's sibling study on
  [408-three-black-crows](../../408-three-black-crows/) documents for the bearish
  three-candle pattern next door. Jegadeesh (1990, *Evidence of predictable behavior of
  security returns*, JF) and Lehmann (1990, *Fads, martingales, and market efficiency*, QJE)
  document short-term reversal in individual stocks — the headwind a naive post-star short
  faces.
- **The unconditional drift.** Equities drift **up** on average, so any *short* starts with
  a base-rate headwind; a signed-short return that is negative but statistically
  indistinguishable from the basket's own unconditional short-of-everything mean is **not**
  evidence of a bearish edge — it is the market's ordinary drift. This is why the study's
  certifying statistic is a **Welch *t* against the unconditional base**, not a one-sample
  *t* against zero (see [`strategy.py`](../evening_star/strategy.py) and the worked example
  in the synthetic control, where a **known** zero-edge panel still shows a "significant"
  vs-zero *t* purely from its own embedded drift).

## The broad evidence on candlestick patterns

- **Marshall, Young & Rose (2006), *Candlestick technical trading strategies: Can they
  create value for investors?* (Journal of Banking & Finance)** — test the full candlestick
  taxonomy (including the morning/evening star family) on DJIA components and find **no
  value** after accounting for data-snooping. Our large-cap null is consistent with this.
- **Horton (2009)** and **Marshall, Young & Cahan (2008)** extend the null to other markets.
- **Lo, Mamaysky & Wang (2000), *Foundations of technical analysis* (JF)** — a careful
  kernel-smoothing study that finds *some* chart patterns carry marginal information; the
  star-reversal family is not among the clean survivors once costs and multiple testing are
  charged.

## Why a high event count still needs a placebo + HAC + a Bonferroni correction

- **Newey-West (1987) HAC** standard errors for the signed-short mean (event ordering can
  carry serial dependence within a name; see [`strategy.hac_t`](../evening_star/strategy.py)).
  Reported as an *informational* cross-check — the decisive number is the **Welch *t***
  against the basket's own unconditional pool, since a plain HAC-vs-zero test is
  contaminated by the tape's ordinary up-drift (see above).
- **Coin-flip label-shuffle placebo** (Fisher's randomization logic; Efron & Tibshirani, *An
  Introduction to the Bootstrap*, 1993) — draw the same event count from the unconditional
  pool, sign each by a fair coin, and ask how often a random pick beats the observed mean.
  See [`strategy.placebo_pvalue`](../evening_star/strategy.py).
- **Bonferroni across the basket.** With 30 tickers tested independently for a per-name
  version of the same effect, the two-sided significance bar must widen to
  `|t| >= z(1 - 0.025/30) ~= 3.14` (see
  [`strategy.bonferroni_z`](../evening_star/strategy.py) and
  [`strategy.per_ticker_stats`](../evening_star/strategy.py)) — otherwise the single
  loudest name in a 30-name basket gets mistaken for "the" signal. Harvey, Liu & Zhu (2016,
  *…and the Cross-Section of Expected Returns*, RFS) and White (2000, *A reality check for
  data snooping*, Econometrica) motivate the general principle.

## Method lineage (the desk's shared engine)

- **Precise OHLC detector + signed event study.**
  [`strategy.is_evening_star`](../evening_star/strategy.py) and
  [`strategy.collect_events`](../evening_star/strategy.py) — the textbook Nison-style
  evening star (tall body, small gapping star, ≥50% penetration back into the first
  candle), signed short, with one execution lag.
- **Drift-neutral inference.** [`strategy.welch_t`](../evening_star/strategy.py) (the
  decisive comparison), [`strategy.hac_t`](../evening_star/strategy.py) (informational,
  vs zero), coin-flip placebo, and the Bonferroni bar.
- **Deterministic synthetic control.**
  [`data.synthetic_panel`](../evening_star/data.py) plants a known post-pattern crash on a
  panel with its own embedded up-drift (mirroring the real basket's own contamination
  risk); with the edge set to zero the drift-neutral inference must NOT manufacture
  significance (checked over 20 seeds) — the offline core runs with no network.

## Data sources used here

- **yfinance** daily OHLCV (`auto_adjust=True`) for a fixed 30-name liquid large-cap + SPY
  basket, 2005-01-03 → 2026-06-30, cached under `_cache/es_*.parquet`. All headline numbers
  are pinned in [`docs/results.md`](results.md) (fingerprints `bde089f6e8ee` / `97e1b85d26c3`)
  and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- **[186-morning-star](../../186-morning-star/)** — the **bullish mirror**. Also runs an
  evening-star arm internally (a looser detector, a random-day-with-direction control arm,
  15 tickers, only 1-day/5-day horizons) and lands on the same qualitative answer (no
  certifiable evening-star excess, *t* = −0.18 there). This study is the **dedicated**
  bearish-side teardown: a wider 30-name basket, three horizons (1/5/10d) *per the brief*,
  an unconditional-base benchmark instead of a random-day-with-direction control, a
  strict-gap and prior-uptrend myth check, a **Bonferroni correction across the basket**
  (per-ticker, not just pooled), and an explicit short-timer with a cost + borrow sweep.
  Independent methodology, same conclusion — the honest kind of replication.
- **[404-shooting-star](../../404-shooting-star/)** — a **single-candle** bearish reversal
  (long upper wick, small body, little lower wick) after an uptrend. No "star" gap, no
  three-candle structure — a different pattern entirely, sharing only the bearish-top
  narrative and the name "star."
- **[408-three-black-crows](../../408-three-black-crows/)** — **three consecutive** tall
  red candles (no star, no gap requirement) — the mechanism this study shares (a
  pre-selected-down-day is a headwind for a fresh short), but a structurally different
  candle shape.
- **[402-engulfing-pattern](../../402-engulfing-pattern/)** — a **two-candle** reversal
  (the second candle's body engulfs the first's). Different candle count, different
  geometry, the same event-study harness lineage.

None of the siblings run the dedicated three-candle, gap-and-penetration evening-star
detector against an unconditional-base, Bonferroni-corrected, cost-and-borrow-charged
bar — this study's own axis.
