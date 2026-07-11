# References & literature map — Study 694 (Matching Low)

## The claim under test

- **The folklore.** The **matching low** is one of the simpler named candlestick
  reversal figures: two consecutive **down** (black) candles whose **closes land at
  essentially the same price** — "a double-bottom in miniature." The reading: the market
  tested the exact same price twice and twice failed to break it, so that price is
  support, and the decline should **reverse upward**. Unlike most of the desk's
  candlestick-pattern studies, the matching low makes **no short-side claim** — it is a
  purely bullish, long-only signal.
- **The canonical source.** Steve Nison, *Japanese Candlestick Charting Techniques*
  (1991/2001, New York Institute of Finance) names and codifies the matching low among
  the minor reversal figures. Thomas Bulkowski's *Encyclopedia of Candlestick Charts*
  (2008, Wiley) catalogs it with his own historical hit-rate screens; this study measures
  it independently, on its own basket, protocol and horizon set, and does not borrow his
  numbers. Modern restatements appear on Investopedia, StockCharts ChartSchool and most
  price-action courses, usually alongside its cousin the **tweezer bottom** (see the
  dedup map below).
- **What we test.** Whether entering long the next morning after a confirmed matching
  low earns a positive (and significant) return over the next 1/5/10/20 days — measured
  *fairly* against what the same basket earns on an unconditional day (not against zero,
  which would just reward the basket's ordinary positive drift, present with or without
  any pattern).

## Why "matching closes" needs an honest tolerance, stated up front

- **Real closes almost never tie to the tick.** A scanner enforcing an exact match would
  find almost nothing on continuous, high-precision daily closes. We use a **loose,
  realistic tolerance** — closes within **0.15%** of each other — as the default detector,
  and a **strict, near-exact** tolerance (**0.03%**) as a myth-check, so the choice of
  tolerance is a transparent, tested decision rather than a hidden knob (see
  [`strategy.py`](../matching_low/strategy.py), `DEFAULT_TOL` / `STRICT_TOL`).
- **The pattern is long-only.** Because there is no short-side mirror, the honest
  benchmark is the basket's own **unconditional** forward-return pool (what an
  always-long trader earns on the same names/window) — not a comparison to zero, and not
  a direction-matched mix like the desk's two-flavor patterns (e.g.
  [693-tasuki-gap](../../693-tasuki-gap/)) need.
- **A downtrend-context myth-check.** The classic reading implies the "support" only
  means something if the market was genuinely declining into the pattern — two red days
  in an uptrend's minor pullback are not the same claim. We test a filter requiring a
  real prior downtrend (see `strategy._prior_downtrend`) and report whether it sharpens
  or dilutes the effect.

## The broad evidence on candlestick patterns

- **Marshall, Young & Rose (2006), *Candlestick technical trading strategies: Can they
  create value for investors?* (Journal of Banking & Finance)** — test the full
  candlestick taxonomy on DJIA components and find **no value** after accounting for
  data-snooping. Our large-cap null is consistent with this.
- **Horton (2009)** and **Marshall, Young & Cahan (2008)** extend the null to other
  markets and other named reversal/continuation figures.
- **Lo, Mamaysky & Wang (2000), *Foundations of technical analysis* (JF)** — a careful
  kernel-smoothing study that finds *some* chart patterns carry marginal information;
  simple two-candle reversal shapes are not among the clean survivors once costs and
  multiple testing are charged.

## Why a common pattern still needs a placebo + HAC + a Bonferroni correction

- **Newey-West (1987) HAC** standard errors for the forward-return mean (event ordering
  can carry serial dependence within a name; see
  [`strategy.hac_t`](../matching_low/strategy.py)). Reported as an *informational*
  cross-check — the decisive number is the **Welch *t*** against the unconditional pool,
  since a plain HAC-vs-zero test is contaminated by the tape's ordinary drift.
- **Random-draw placebo** (Fisher's randomization logic; Efron & Tibshirani, *An
  Introduction to the Bootstrap*, 1993) — draw the same event count from the
  unconditional pool and ask how often a random pick beats the observed mean. See
  [`strategy.placebo_pvalue`](../matching_low/strategy.py).
- **Bonferroni across the basket.** With 30 tickers carrying >= 6 events tested
  independently for a per-name version of the same effect, the two-sided significance
  bar must widen to `|t| >= z(1 - 0.025/30) ~= 3.14` (see
  [`strategy.bonferroni_z`](../matching_low/strategy.py) and
  [`strategy.per_ticker_stats`](../matching_low/strategy.py)) — otherwise the single
  loudest name in the basket gets mistaken for "the" signal. Harvey, Liu & Zhu (2016,
  *…and the Cross-Section of Expected Returns*, RFS) and White (2000, *A reality check
  for data snooping*, Econometrica) motivate the general principle.

## Method lineage (the desk's shared engine)

- **Precise two-bar detector + event study.**
  [`strategy.matching_low_signal`](../matching_low/strategy.py) and
  [`strategy.collect_events`](../matching_low/strategy.py) — two down candles, closes
  matching within a stated tolerance, one execution lag.
- **Drift-neutral inference vs the unconditional base.**
  [`strategy.welch_t`](../matching_low/strategy.py) (the decisive comparison),
  [`strategy.hac_t`](../matching_low/strategy.py) (informational, vs zero), the
  random-draw placebo, and the Bonferroni bar — the same engine lineage as
  [693-tasuki-gap](../../693-tasuki-gap/), [689-upside-gap-two-crows](../../689-upside-gap-two-crows/)
  and [683-evening-star](../../683-evening-star/).
- **Deterministic synthetic control.**
  [`data.synthetic_panel`](../matching_low/data.py) plants a known post-pattern
  **reversal** on a panel with its own embedded drift; with the edge set to zero the
  drift-neutral inference must NOT manufacture significance (checked over 20 seeds) — the
  offline core runs with no network.

## Data sources used here

- **yfinance** daily OHLCV (`auto_adjust=True`) for a fixed 30-name liquid large-cap +
  SPY basket, 2005-01-03 -> 2026-06-30, cached under `_cache/ml_*.parquet`. All headline
  numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- **[409-tweezer-tops-bottoms](../../409-tweezer-tops-bottoms/)** — the closest
  relative: two candles with **matching lows** (the *low* of the wick, not the close) at
  a bottom, no requirement that either candle even be down. The matching low's whole
  identity is the **repeated close**, on **two down candles specifically** — a stricter,
  body-level claim than the tweezer's wick-level one. The two figures often co-occur on
  the same two-bar pair but are testing different geometry (close-equality vs
  low-equality) and this study runs neither pattern's detector for the other.
- **[460-counterattack-lines](../../460-counterattack-lines/)** — also an "equal close"
  reversal idiom, but the counterattack line requires the **second candle to be the
  opposite color** of the first (a down candle then an up candle closing at the same
  price, or the bearish mirror) — the reversal is asserted to already be visible *within*
  the pattern. The matching low requires **both** candles to be the *same* color (down,
  down) with the reversal only implied for *afterward* — a fundamentally different
  two-candle shape from the counterattack line's opposite-color pair.
- **[696-double-bottom](../../696-double-bottom/)** — the macro version of the same
  "tested the same price twice" idea, but built from a **swing-level chart pattern**
  spanning many bars/weeks (two prominent troughs separated by an intermediate peak),
  not a strict two-*candle* micro pattern confirmed the next session. This study is the
  miniature, next-day version; 696 is the multi-week pattern-recognition problem.

None of the siblings run this study's specific two-consecutive-down-candles,
matching-**close**, long-only, Bonferroni-corrected, cost-and-tolerance-tested bar — the
matching low is this study's own axis.
