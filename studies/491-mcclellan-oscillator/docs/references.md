# References & literature map — Study 491 (McClellan Oscillator)

## The claim under test

- **The folklore.** Take the daily **net advances** (advancing issues − declining issues)
  across the market and compute a fast-minus-slow pair of exponential moving averages:
  **McOsc = EMA19(net-adv) − EMA39(net-adv)** (the classic 10% / 5% smoothing constants).
  The oscillator is read as **breadth momentum**: when it crosses **up through zero from
  negative**, breadth has turned bullish and "the index is about to rally." This is a
  retail/technician staple, built into StockCharts, TradingView (`$NYMO`), MetaStock and
  every breadth dashboard.
- **The source.** **Sherman & Marian McClellan** introduced the oscillator in **1969**
  (popularised through their booklet *Patterns for Profit*, 1970, and the McClellan
  Financial Publications newsletter). The companion **McClellan Summation Index** is the
  running cumulative sum of the oscillator. The primary lineage is the McClellans' own work;
  modern restatements appear in StockCharts' ChartSchool, John Murphy's *Technical Analysis
  of the Financial Markets*, and Investopedia.
- **Breadth-momentum cousins.** The Arms Index / TRIN (Study 490), the advance-decline line
  (Study 188-family), and the high-low index are affine relatives — all summarise the same
  net-breadth input. They inherit the same drift confound tested here.

## Why this is a proxy study (and what that caps)

The genuine oscillator is built on **exchange** advance/decline counts (thousands of NYSE
issues). With no offline exchange-breadth feed, we approximate net advances with the sign of
the daily close-to-close move across a small ETF basket. A 5–10 name basket is a *coarse,
noisy* estimate of true breadth:

- This makes the test **conservative against a false positive** — coarse breadth can only
  blur a real signal, never manufacture one. So a *null* result on the proxy is weak evidence
  the true indicator is also null, **but** the headline here is stronger than null: the
  trigger is *significantly worse* than random, which a richer basket is very unlikely to
  reverse.
- A future run can widen the basket to the sector ETFs `XLK XLF XLE XLV XLI XLY XLP XLU XLB`
  (cached on demand via `data.breadth_members(allow_fetch=True)`), or splice a real
  `$NYAD`/`$NYMO` series, to tighten the estimate.

## Why the high one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample
  *t* of a long-only entry rule against **zero** measures that drift, not the rule (here the
  60-day one-sample *t* = +3.01 is *entirely* beta — the trigger still **loses** to random
  by 100 bps). The desk's standing rule is *signal-vs-baseline*, never *signal-vs-zero*.
- **Data snooping on chart/breadth tools.** Lo, Mamaysky & Wang (2000, *Foundations of
  Technical Analysis*, Journal of Finance) formalise testing technical signals against a
  properly matched null; Sullivan, Timmermann & White (1999, *Data-Snooping, Technical
  Trading Rule Performance, and the Bootstrap*, JF) and White (2000, *A Reality Check for
  Data Snooping*, Econometrica) show how trend-fitted rules manufacture significance unless
  raced against a fair benchmark.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch
  (1947) two-sample *t* for the trigger-vs-random difference.

## Method lineage (the desk's shared engine)

- **Causal oscillator + up-cross trigger.** [`strategy.mcclellan`](../mcclellan_oscillator/strategy.py),
  [`strategy.up_cross_dates`](../mcclellan_oscillator/strategy.py) — forward-only EMAs, the
  cross read on the close of *t*, entry at *t+1*.
- **Forward-return + HAC t + random baseline.** [`strategy.forward_returns`](../mcclellan_oscillator/strategy.py),
  [`strategy.hac_t`](../mcclellan_oscillator/strategy.py), [`strategy.run_experiment`](../mcclellan_oscillator/strategy.py).
- **Geometry placebo.** [`strategy.shuffled_breadth_placebo`](../mcclellan_oscillator/strategy.py) —
  time-permute the net-advances series, keep its marginal, rebuild the oscillator.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../mcclellan_oscillator/data.py)
  plants a real post-cross bounce (knob `edge`); with `edge = 0` the detector must NOT
  manufacture significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. The breadth proxy is derived from the same cached basket. All headline numbers
  are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../../490-arms-index-trin`](../../490-arms-index-trin) — the sibling breadth indicator
  (TRIN), same net-breadth input, tested with the same random-entry idiom.
- [`../../450-andrews-pitchfork`](../../450-andrews-pitchfork) — the template study; same
  random-entry-baseline + geometry-placebo design.
- The broader technical-indicator zoo (CCI, ADX, Aroon, Coppock …) and the
  **research-method demos** (data-mining-roulette, look-ahead, curve-fitting) frame why a
  signal-vs-zero *t* is not enough; the McClellan oscillator is another clean live example
  of beta masquerading as a forecast.
