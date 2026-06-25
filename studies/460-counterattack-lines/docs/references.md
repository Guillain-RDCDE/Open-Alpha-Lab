# References & literature map — Study 460 (Counterattack / Meeting Lines)

## The claim under test

- **The folklore.** A *counterattack line* (a.k.a. **meeting line**) is a two-candle reversal
  pattern. The **bullish** version, the one tested here, appears after a downtrend: a long
  **black** (down) candle, then a **white** (up) candle that gaps lower on the open but rallies
  all session to **close at ~the same price as the prior close** — the two closes "meet". The
  lore says this equal-close meeting marks where the bears lost control: a **buy / reversal up**.
  It is the weaker cousin of the *piercing line* (which must close *above* the prior candle's
  midpoint); the counterattack only requires the closes to meet.
- **The source.** **Steve Nison**, *Japanese Candlestick Charting Techniques* (1991; 2nd ed.
  2001), introduced the Western world to *deai sen* / *gyakushu sen* ("meeting" / "counterattack"
  lines) drawn from the Japanese candlestick tradition (Munehisa Homma lineage, Sakata rules).
  Greg Morris (*Candlestick Charting Explained*, 1992/2006) and Bulkowski's
  *Encyclopedia of Candlestick Charts* (2008) catalogue and rank the pattern; TradingView,
  StockCharts ChartSchool and most charting suites restate the rule.
- **Variants.** The *piercing line* and *dark-cloud cover* are the stronger close-above /
  close-below relatives; the *bullish/bearish engulfing* and *tweezer* patterns are nearby
  two-candle reversals. All are **two-bar close-geometry patterns after a trend** and inherit
  the same drift confound tested here.

## Why this is a mechanical-proxy study

Candlestick recognition is *semi-subjective* — "long" candle, "gap", "same close" are eyeballed.
Following the desk's design for this kind, we encode the **tightest mechanical rule a proponent
would accept** and state the irreducible thresholds explicitly:

- **Objective context.** Down leg = close[t-1] strictly below close[t-10] (a confirmed prior
  decline, no future bars).
- **Objective candles.** Candle t-1 black (close < open), candle t white (close > open), open[t]
  gaps below close[t-1].
- **Objective meeting.** |close[t] − close[t-1]| / close[t-1] ≤ 15 bps — a concrete equal-close
  tolerance instead of an eyeballed "same level".
- **The honest baseline.** The only meaningful comparison on an upward-drifting index is the
  **random-entry** control (same instrument, epoch and hold), because *any* dip-buy inherits the
  drift. We add a **close-scramble placebo** that keeps the down-leg + gap context but ignores
  the equal-close test — the direct test of "does the *meeting* matter, or just the dip?"

## Why the high one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample
  *t* of a long-only entry rule against **zero** measures that drift, not the rule. See Fama &
  French on the equity premium; the desk's standing rule is *excess-vs-excess* and
  *signal-vs-baseline*, never *signal-vs-zero*.
- **Data snooping on chart patterns.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis: Computational Algorithms, Statistical Inference, and Empirical Implementation*, JF)
  formalize testing chart patterns against a properly matched null and find most carry little
  *conditional* information. Sullivan, Timmermann & White (1999, *Data-Snooping, Technical
  Trading Rule Performance, and the Bootstrap*, JF) and White (2000, *A Reality Check for Data
  Snooping*, Econometrica) show how rules selected from a large family manufacture significance
  unless raced against a fair benchmark. Marshall, Young & Rose (2006, *Candlestick technical
  trading strategies: Can they create value for investors?*, JBF) find candlestick signals add
  no value after proper benchmarking — directly on point for this pattern.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the meeting-vs-random difference.

## Method lineage (the desk's shared engine)

- **Mechanical pattern detection.** [`strategy._meeting_mask`](../counterattack_lines/strategy.py),
  [`strategy.meeting_entries`](../counterattack_lines/strategy.py) — the down-leg + opposite-colour
  + gap + equal-close rule with the timing lag baked in.
- **Forward-return + HAC t + random baseline.** [`strategy.forward_returns`](../counterattack_lines/strategy.py),
  [`strategy.hac_t`](../counterattack_lines/strategy.py), [`strategy.run_experiment`](../counterattack_lines/strategy.py).
- **Geometry placebo.** [`strategy.close_scramble_placebo`](../counterattack_lines/strategy.py) —
  keep the down-leg/gap candidate pool, ignore the equal-close meeting.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../counterattack_lines/data.py)
  plants a real meeting-line bounce (knob `edge`); with `edge = 0` the detector must NOT
  manufacture significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) OHLC for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced
  by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../450-andrews-pitchfork`](../450-andrews-pitchfork) — the sibling "price respects the
  geometry" candlestick/chart-tool teardown that this study's harness mirrors.
- [`../../402-...`](../../) through the candlestick zoo — most two-bar reversal patterns land
  None × Mirage for the same reason: a pattern read off recent price re-describes the dip.
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting) frame why a
  signal-vs-zero *t* is not enough; the counterattack line is a clean live example of a down-leg
  dip-buy (beta) wearing a candlestick costume.
