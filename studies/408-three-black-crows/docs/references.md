# References & literature map — Study 408 (Three Black Crows)

## The claim under test

- **The folklore.** "Three black crows is a **bearish reversal**: after a rally, three long
  red candles in a row — each opening inside the previous body and closing near the low — show
  the bulls have lost control and a **crash / sell-off is just beginning**. Sell or short." It
  is one of the most recognizable candlestick patterns in technical analysis.
- **The canonical source.** Steve Nison, *Japanese Candlestick Charting Techniques* (1991,
  New York Institute of Finance) — the book that introduced candlestick analysis to Western
  traders — codifies the three-black-crows (and its bullish twin, three white soldiers) as a
  reversal signal. Repeated verbatim by Investopedia ("Three Black Crows Definition"),
  StockCharts, and most charting platforms.
- **What we test.** Whether shorting the precise OHLC pattern, entered one day after the
  confirming close, earns a positive (and significant) return over the next 1/3/5/10 days —
  i.e. whether the "crash" actually arrives.

## Why the bearish reading is mechanically suspect

- **Selection on a fall.** Three consecutive red bodies pre-select stocks that have *already*
  dropped. Short-horizon **mean reversion** then tends to dominate (a bounce), which puts the
  naive bearish bet on the wrong side. Jegadeesh (1990, *Evidence of predictable behavior of
  security returns*, JF) and Lehmann (1990, *Fads, martingales, and market efficiency*, QJE)
  document short-term reversal in individual stocks — exactly the headwind a post-crows short
  faces.
- **The unconditional drift.** Equities drift **up** on average, so any *short* starts with a
  base-rate headwind; the honest null is a zero signed-short return, and we also report the
  unconditional (long) base rate alongside each horizon.

## The broad evidence on candlestick patterns

- **Marshall, Young & Rose (2006), *Candlestick technical trading strategies: Can they create
  value for investors?* (Journal of Banking & Finance)** — test the full candlestick taxonomy
  (including three black crows / three white soldiers) on DJIA components and find **no value**
  after accounting for the data-snooping. Our large-cap null is consistent with this.
- **Horton (2009)** and **Marshall, Young & Cahan (2008)** extend the null to other markets.
- **Lo, Mamaysky & Wang (2000), *Foundations of technical analysis* (JF)** — a careful kernel-
  smoothing study that finds *some* chart patterns carry marginal information; three black
  crows is not among the survivors once costs and multiple testing are charged.

## Why a high event count still needs a placebo + HAC

- **Newey-West (1987) HAC** standard errors for the signed-short mean against zero (event
  ordering can carry serial dependence within a name). See
  [`strategy.hac_t`](../three_black_crows/strategy.py).
- **Coin-flip label-shuffle placebo** (Fisher's randomization logic; Efron & Tibshirani, *An
  Introduction to the Bootstrap*, 1993) — draw the same event count from the unconditional
  pool, sign each by a fair coin, and ask how often a random pick beats the observed mean. See
  [`strategy.placebo_pvalue`](../three_black_crows/strategy.py). Here the real signal lands in
  the *left* tail (worse than random).
- **Selection on a famous rule.** Harvey, Liu & Zhu (2016, *…and the Cross-Section of Expected
  Returns*, RFS) and White (2000, *A reality check for data snooping*, Econometrica) motivate
  charging a multiple-testing/selection penalty before believing any single pattern — moot here
  since the raw effect is already a null in the wrong direction.

## Method lineage (the desk's shared engine)

- **Precise OHLC detector + signed event study.**
  [`strategy.is_three_black_crows`](../three_black_crows/strategy.py) and
  [`strategy.collect_events`](../three_black_crows/strategy.py) — the textbook real-body crow,
  signed short, with one execution lag.
- **HAC / one-sample t + coin-flip placebo.**
  [`strategy.hac_t`](../three_black_crows/strategy.py),
  [`strategy.onesample_t`](../three_black_crows/strategy.py),
  [`strategy.placebo_pvalue`](../three_black_crows/strategy.py).
- **Deterministic synthetic control.**
  [`data.synthetic_panel`](../three_black_crows/data.py) plants a known post-pattern crash;
  with the edge set to zero the inference must NOT manufacture significance — the offline core
  runs with no network.

## Data sources used here

- **yfinance** daily OHLCV (`auto_adjust=True`) for a fixed 30-name liquid large-cap + SPY
  basket, 2005-01-03 → 2026-06-18, cached under `_cache/tbc_*.parquet`. All headline numbers
  are pinned in [`docs/results.md`](results.md) (fingerprint `bf1d6cb7ca54`) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[402-engulfing-pattern](../../402-engulfing-pattern/)** — the engulfing reversal candle,
  the same event-study harness; another candlestick null.
- **[404-shooting-star](../../404-shooting-star/)**, **[405-doji-reversal](../../405-doji-reversal/)**,
  **[186-morning-star](../../186-morning-star/)**, **[187-three-soldiers](../../187-three-soldiers/)** —
  the rest of the desk's candlestick-pattern teardowns (the bullish twin, "three white
  soldiers," lives in 187).
- The **research-method demos** (multiple-testing, data-mining-roulette) frame why a single
  famous pattern needs a placebo + selection penalty before it earns a stamp.
