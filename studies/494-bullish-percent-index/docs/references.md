# References & literature map — Study 494 (Bullish Percent Index)

## The claim under test

- **The folklore.** The Bullish Percent Index (BPI) is a market-breadth oscillator: the
  percentage of a basket whose members sit on a Point & Figure (P&F) *buy* signal, scaled 0-100.
  The teaching: BPI **above 70** is "overbought" (a market-*top* warning, lighten up), BPI
  **below 30** is "oversold" (breadth washed out, a market-*bottom* and a high-probability buy),
  and the *reversal* up out of oversold is the entry. This is a staple of the
  point-and-figure / market-internals tradition.
- **The source.** **A.W. (Abe) Cohen** of Chartcraft / Investors Intelligence introduced the
  Bullish Percent Index in **1955** as a way to read NYSE breadth from P&F signals. **Earl
  Blumenthal** (1975) and **Michael Burke** extended the bull/bear-alert reversal rules. The
  modern canonical write-up is **StockCharts' ChartSchool** ("Bullish Percent Index"); Thomas
  Dorsey's *Point & Figure Charting* (1995) popularized it for a generation of technicians.
- **Variants.** Sector BPIs ($BPENER, $BPINFO, …), NYSE/Nasdaq composite BPIs, and the related
  "% above 50/200-day MA" breadth indices ($SPXA50R) are all the same idea — count participating
  members — and inherit the same drift confound tested here.

## Why this is a "theory" / mechanical-proxy study

True BPI requires per-member Point & Figure box/reversal bookkeeping over a *full exchange*.
Following the desk's design for breadth folklore, we encode the **tightest transparent
mechanical proxy a proponent would accept** and state the irreducible approximation explicitly:

- **Objective breadth.** The **percentage of the basket trading above its 50-day SMA** — the
  standard, reproducible stand-in for the P&F-buy-signal count (the two track each other
  closely; both ask "is this member in an uptrend?"). The SMA is causal: every vote uses only
  data through bar *t*, no look-ahead.
- **Objective entry.** A long fires on the bar where BPI **crosses up** through 30 (Cohen's
  "reversal into a column of X's out of oversold"); entry at the **next close** (one lag). We
  also report the simpler "BPI < 30" level rule.
- **The honest baseline.** The only meaningful comparison on an upward-drifting index is the
  **random-entry** control (same instrument, epoch and hold), because *any* dip-buy inherits the
  drift. We add a **scrambled-breadth placebo** (block-shuffle the BPI in time) that destroys the
  breadth-to-price timing while keeping the marginal — the direct test of "does the timing
  matter?"

The proxy is coarse and **caps** the test; a P&F-exact BPI on full exchange data could differ in
detail, but the drift confound — the reason the apparent edge appears — is structural and would
survive a finer proxy.

## Why the high one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample
  *t* of a long-only entry rule against **zero** measures that drift, not the rule. See Fama &
  French on the equity premium; the desk's standing rule is *signal-vs-baseline*, never
  *signal-vs-zero*. (Here even the one-sample *t* barely registers — the cross fires too rarely
  to bank much drift — which makes the failure-vs-random doubly clear.)
- **Data snooping on chart tools.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis*, JF) formalize testing chart/breadth rules against a properly matched null;
  Sullivan, Timmermann & White (1999, *Data-Snooping, Technical Trading Rule Performance, and
  the Bootstrap*, JF) and White (2000, *A Reality Check for Data Snooping*, Econometrica) show
  how trend-fitted rules manufacture significance unless raced against a fair benchmark.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the cross-vs-random difference.

## Method lineage (the desk's shared engine)

- **Breadth oscillator (BPI proxy).** [`strategy.bpi`](../bullish_percent_index/strategy.py) —
  causal % above the 50-day SMA across the basket.
- **Oversold-cross / level entries.** [`strategy.oversold_cross_entries`](../bullish_percent_index/strategy.py),
  [`strategy.oversold_level_entries`](../bullish_percent_index/strategy.py).
- **Forward-return + HAC t + random baseline.** [`strategy.forward_returns`](../bullish_percent_index/strategy.py),
  [`strategy.hac_t`](../bullish_percent_index/strategy.py), [`strategy.run_experiment`](../bullish_percent_index/strategy.py).
- **Breadth-timing placebo.** [`strategy.scrambled_breadth_placebo`](../bullish_percent_index/strategy.py) —
  block-shuffle BPI in time, keep the marginal.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../bullish_percent_index/data.py)
  plants a real oversold-bounce (knob `edge`); with `edge = 0` the detector must NOT manufacture
  significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes; SPY is the traded tape and the breadth
  basket is SPY, QQQ, IWM, DIA, 2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June
  dropped), cached as parquet under `_cache/`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../450-andrews-pitchfork`](../450-andrews-pitchfork) — the template: a chart tool that turns
  out to be beta-in-a-costume, tested with the identical random-entry + placebo idiom.
- [`../../178-cci`](../../178-cci) and the broader technical-indicator zoo — most land
  None × Mirage for the same reason: an indicator fitted to past price re-describes the trend.
- The market-internals / breadth siblings (advance-decline, % above moving average) and the
  **research-method demos** (data-mining-roulette, look-ahead, curve-fitting) frame why a
  signal-vs-zero *t* is not enough; the BPI is a clean live example of breadth drift masquerading
  as a turn forecast.
