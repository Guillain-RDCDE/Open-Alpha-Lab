# References & literature map — Study 486 (Gann Hi-Lo Activator)

## The claim under test

- **The folklore.** The *Gann Hi-Lo Activator* is a trailing stop-and-reverse line: a simple
  moving average of recent **highs** and of recent **lows** that **flips** with price. While the
  regime is long the activator tracks the SMA of lows (a stop *below* price); a close *below* it
  flips the regime short and the activator jumps to the SMA of highs (a stop *above*). The
  technician's claim is that the **flip forecasts trend** — a flip up (close above the activator)
  signals the start of a new up-leg, so it's a high-odds **buy**; a flip down a sell. It is built
  into MetaTrader, TradingView, NinjaTrader and Thinkorswim and is a staple of Gann-trading
  write-ups.
- **The source.** The indicator is attributed to **Robert Krausz**, who popularised it in his
  *W. D. Gann Treasure Discovered* / *New Gann Swing Chartist* work (Fibonacci Trader, 1990s),
  building on **W. D. Gann's** original swing-chart and high/low-activation ideas (1930s–40s).
  Modern restatements appear in Investopedia, StockCharts' ChartSchool, and the platform docs for
  the "Gann HiLo Activator" study. The mechanics (SMA-of-highs / SMA-of-lows flip line) are
  identical across these sources; only the default period varies (commonly 3, 10, or 14).
- **Variants.** Different period lengths, EMA instead of SMA, and "HiLo bands" (plot both lines
  rather than the single flip line) are **affine/parametric variants of the same construction**
  and inherit the same drift confound tested here.

## Why this is a "theory" / mechanical-proxy study

The Gann Hi-Lo Activator is fully mechanical (no eyeballing) — but proponents add discretion in
the period choice and in combining it with other Gann tools. Following the desk's design we
encode the **tightest mechanical rule a proponent would accept** and state the parameter choice
explicitly:

- **Objective flip.** Period-10 SMA of highs / lows, both **shifted one bar** so the line at bar
  *t* uses only data through *t-1*; the flip is read on the close of *t* — a documented one-bar
  discipline, no look-ahead.
- **Objective entry.** First bar of each short→long flip; entered at the **next close** (one
  documented lag).
- **The honest baseline.** The only meaningful comparison on an upward-drifting index is the
  **random-entry** control (same instrument, epoch and hold), because *any* long-only entry
  inherits the drift. We add a **shuffled-flip timing placebo** that keeps the number of flips and
  the price marginal but moves the flip dates at random — the direct test of "does the flip's
  *timing* matter?"

## Why the high one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample *t*
  of a long-only entry rule against **zero** measures that drift, not the rule. See Fama & French
  on the equity premium; the desk's standing rule is *excess-vs-excess* and *signal-vs-baseline*,
  never *signal-vs-zero*. Here every horizon clears one-sample *t* = 2.6–6.2 yet **none** beats a
  random-day entry — a textbook beta mirage.
- **Data snooping on chart tools.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis*, *Journal of Finance*) formalize testing chart patterns against a properly matched
  null; Sullivan, Timmermann & White (1999, *Data-Snooping, Technical Trading Rule Performance,
  and the Bootstrap*, *Journal of Finance*) and White (2000, *A Reality Check for Data Snooping*,
  *Econometrica*) show how trend-fitted rules manufacture significance unless raced against a fair
  benchmark.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the flip-vs-random difference.

## Method lineage (the desk's shared engine)

- **Flipping activator + flip-up entry.** [`strategy.hilo_activator`](../gann_hilo_activator/strategy.py),
  [`strategy.flip_up_entries`](../gann_hilo_activator/strategy.py) — the mechanical SMA-of-highs /
  SMA-of-lows flip with the one-bar shift baked in.
- **Forward-return + HAC t + random baseline.** [`strategy.forward_returns`](../gann_hilo_activator/strategy.py),
  [`strategy.hac_t`](../gann_hilo_activator/strategy.py), [`strategy.run_experiment`](../gann_hilo_activator/strategy.py).
- **Timing placebo.** [`strategy.shuffled_flip_placebo`](../gann_hilo_activator/strategy.py) —
  keep the flip count and price marginal, scramble the flip dates.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../gann_hilo_activator/data.py)
  plants a real post-flip trend (knob `edge`); with `edge = 0` the detector must NOT manufacture
  significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../../450-andrews-pitchfork`](../../450-andrews-pitchfork) — the same "price respects the
  channel/line" folklore tested with the random-entry baseline + geometry placebo idiom; the
  direct sibling of this study.
- [`../../117-supertrend`](../../117-supertrend) and [`../../116-faber-timing`](../../116-faber-timing)
  — other trailing-stop / trend-flip rules; the broader technical-indicator zoo mostly lands
  None × Mirage because an indicator fitted to past price re-describes the trend.
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting) frame why a
  signal-vs-zero *t* is not enough; the Gann Hi-Lo Activator is a clean live example of beta
  masquerading as a flip signal.
