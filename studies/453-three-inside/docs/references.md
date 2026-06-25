# References & literature map — Study 453 (Three-Inside-Up / Down)

## The claim under test

- **The folklore.** The **three-inside-up** is a three-candle bullish reversal: (A) a strong
  **down** candle ending a downtrend, (B) a **harami** — an *inside* bar whose body/range sits
  inside A's body — and (C) a **confirmation** candle that closes back **above A's open** (and
  above B's close). The mirror **three-inside-down** is the bearish reversal at a top. The lore
  taught in every candlestick text: the confirming third candle "flips the trend", so a long after
  C (short after the bearish mirror) is a high-probability reversal trade. It is the formalised,
  confirmed cousin of the bare two-candle *harami*.
- **The source.** Japanese candlestick reversal patterns were brought to the West by **Steve
  Nison**, *Japanese Candlestick Charting Techniques* (1991) — the canonical reference for the
  harami and its three-candle confirmations. **Gregory L. Morris**, *Candlestick Charting Explained*
  (1992/2006) catalogues the three-inside-up/down explicitly and is the usual primary source for the
  pattern's exact definition. Thomas Bulkowski's *Encyclopedia of Candlestick Charts* (2008) reports
  empirical hit-rates and is the most-cited "evidence" the believers point to.
- **The thesis we put on trial.** The distinguishing feature versus the bare harami is the
  **confirmation candle**. The headline question for this study is therefore not merely "does the
  pattern work?" but **"does the confirmation candle add edge?"** — the third candle is the part
  the folklore prizes, so it is the part we isolate and test (the harami-only placebo).

## Why this is a mechanical-proxy study

A discretionary trader eyeballs "a confirming close" and "a downtrend". Following the desk's design
for chart patterns, we encode the **tightest mechanical rule a proponent would accept** and state
the irreducible choices explicitly:

- **Objective harami.** Bar B's whole range must sit inside Bar A's *body* (high ≤ body-high,
  low ≥ body-low) — the strict body-harami, the charitable reading.
- **Objective confirmation.** Bar C must close above Bar A's open and above Bar B's close — the
  "closes back past the first candle" rule, read only on closed bars.
- **Objective downtrend.** Bar A's close below the close `trend_lookback` (=5) bars earlier.
- **No look-ahead.** The triplet completes on the close of C (*t*); the long is entered at the
  **next close** (*t+1*).

## Why a high one-sample t (or any absolute return) is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample *t* of
  a long-only entry rule against **zero** measures that drift, not the rule. The desk's standing rule
  is *signal-vs-baseline*, never *signal-vs-zero*. (Here the confirmed rule does not even *win* the
  drift game — it loses to a random-day entry — which is a stronger refutation still.)
- **Data snooping on chart patterns.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis*, *Journal of Finance*) formalise testing chart/candlestick patterns against a properly
  matched null and find most patterns add little once the benchmark is fair. Sullivan, Timmermann &
  White (1999, *Data-Snooping, Technical Trading Rule Performance, and the Bootstrap*, JF) and White
  (2000, *A Reality Check for Data Snooping*, *Econometrica*) show how pattern rules manufacture
  significance unless raced against a fair benchmark. Marshall, Young & Rose (2006, *Candlestick
  technical trading strategies: Can they create value for investors?*, *Journal of Banking &
  Finance*) test candlestick patterns directly and find **no value** — directly on point here.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the confirmed-vs-random and confirmed-vs-harami differences.

## Method lineage (the desk's shared engine)

- **Pattern detection.** [`strategy.three_inside_up`](../three_inside/strategy.py),
  [`strategy.three_inside_entries`](../three_inside/strategy.py) — the mechanical harami +
  confirmation, with the `require_confirm` switch that powers the placebo.
- **Forward-return + HAC t + random baseline.** [`strategy.forward_returns`](../three_inside/strategy.py),
  [`strategy.hac_t`](../three_inside/strategy.py), [`strategy.run_experiment`](../three_inside/strategy.py).
- **Confirmation-candle placebo (the thesis test).**
  [`strategy.harami_only_placebo`](../three_inside/strategy.py) — same harami event, confirmation
  candle dropped; the direct measure of the third candle's marginal contribution.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../three_inside/data.py) plants a
  real post-three-inside-up bounce (knob `edge`); with `edge = 0` the detector must NOT manufacture
  significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) OHLC for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../450-andrews-pitchfork`](../450-andrews-pitchfork) — the sibling chart-tool teardown with the
  same random-entry idiom (there the rule rode the drift; here it can't even do that).
- The candlestick zoo elsewhere on the desk (engulfing, stars, soldiers, NR7 and friends) — most
  land None × Mirage for the same reason: a pattern fitted to recent OHLC re-describes price without
  forecasting it.
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting) frame why an
  absolute return or signal-vs-zero *t* is not enough; the three-inside-up is a clean live example
  where even the celebrated confirmation candle is a *negative* contributor.
