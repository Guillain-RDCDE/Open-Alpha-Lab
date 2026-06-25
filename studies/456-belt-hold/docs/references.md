# References & literature map — Study 456 (Belt-Hold / opening marubozu)

## The claim under test

- **The folklore.** A **bullish belt-hold** (Japanese *yorikiri* — a sumo term, "to force out")
  is a single candle that **opens at its low** (no lower wick) and **closes well up** as a long
  white real body, arriving after a downtrend. Because the open sits at the session extreme,
  buyers are said to have "seized control from the first tick", so the prior down-move reverses —
  a **buy**. The mirror (bearish belt-hold) opens at the high after an uptrend. It is a staple of
  every candlestick site, TradingView's pattern scanner, and broker education pages.
- **The source.** The pattern enters the Western canon through **Steve Nison**, *Japanese
  Candlestick Charting Techniques* (1991) and *Beyond Candlesticks* (1994), which translated the
  Japanese *sakata* candle lore (Munehisa Homma's rice-trading tradition, Edo period) for US
  audiences. Nison classifies the belt-hold (*yorikiri*) as a single-line reversal whose strength
  scales with body length and rarity. Gregory Morris, *Candlestick Charting Explained* (1992/2006)
  catalogues it with the same rule; Bulkowski's *Encyclopedia of Candlestick Charts* (2008)
  tabulates its empirical hit-rate (and finds it unremarkable once base rates are accounted for).
- **Variants.** The "opening marubozu" (open == one extreme, the other end may have a wick) and
  the full "marubozu" (no wicks at all) are the same geometry with tighter wick tolerances; the
  bearish belt-hold is the up-trend mirror. All share the open-at-the-extreme premise tested here.

## Why this is a "mechanical-proxy" study

A discretionary technician eyeballs "opens at the low" and "after a downtrend." Following the
desk's design, we encode the **tightest mechanical rule a proponent would accept** and state the
thresholds explicitly:

- **Objective belt-hold flag.** White body; lower wick ≤ 10% of the bar's high-low range
  (open ≈ low); body ≥ 60% of the range (a tall marubozu); and a **prior downtrend** (close below
  the close 10 bars earlier). All read from the bar completed at the close of *t* — no look-ahead.
- **Objective entry.** Long at the **next** close (one documented lag); hold 5/10/20/60 days.
- **The honest baseline.** The only meaningful comparison on an upward-drifting index is the
  **random-entry** control (same instrument, epoch, hold), because *any* long entry inherits the
  drift. We add a **shape-scramble placebo** that keeps the prior-downtrend filter and the signal
  count but draws from random downtrend bars — the direct test of "does the candle shape matter?"

## Why the high one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample *t*
  of a long-only entry rule against **zero** measures that drift, not the rule. See Fama & French
  on the equity premium; the desk's standing rule is *signal-vs-baseline*, never *signal-vs-zero*.
- **Data snooping on chart patterns.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis*, Journal of Finance) formalize testing chart patterns against a properly matched null
  and find most add little once base rates are respected; Sullivan, Timmermann & White (1999,
  *Data-Snooping, Technical Trading Rule Performance, and the Bootstrap*, JF) and White (2000,
  *A Reality Check for Data Snooping*, Econometrica) show how price-fitted rules manufacture
  significance unless raced against a fair benchmark. A single-candle reversal label is exactly
  the kind of rule that needs a random-entry and a shape placebo to survive.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the belt-hold-vs-random difference.

## Method lineage (the desk's shared engine)

- **Belt-hold detection.** [`strategy.belt_hold_flags`](../belt_hold/strategy.py),
  [`strategy.belt_hold_entries`](../belt_hold/strategy.py) — the open-at-low / tall-body /
  prior-downtrend rule, read on the close of t (one-lag entry).
- **Forward-return + HAC t + random baseline.** [`strategy.forward_returns`](../belt_hold/strategy.py),
  [`strategy.hac_t`](../belt_hold/strategy.py), [`strategy.run_experiment`](../belt_hold/strategy.py).
- **Geometry placebo.** [`strategy.shape_scramble_placebo`](../belt_hold/strategy.py) — keep the
  downtrend filter and the count, scramble the candle shape.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../belt_hold/data.py) plants a
  real post-belt-hold reversal (knob `edge`); with `edge = 0` the detector must NOT manufacture
  significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../../402-doji`](../../402-doji), [`../../403-hammer`](../../403-hammer) and the candlestick
  cluster (402-409) — the same single-candle reversal folklore tested with the random-entry idiom.
- [`../../450-andrews-pitchfork`](../../450-andrews-pitchfork) — the sibling "geometry placebo"
  design (scramble the lines / the candle shape and watch the edge survive).
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting) frame why a
  signal-vs-zero *t* is not enough; the belt-hold is a clean live example of a marginal long-horizon
  edge that turns out to be the downtrend context, not the candle.
