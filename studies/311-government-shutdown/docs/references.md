# References & literature map — Study 311 (Government-Shutdown)

## The claim under test

- **"Buy the dip every time the government shuts down."** A recurring piece of market
  folklore amplified by financial media around every funding fight: *stocks shrug off
  shutdowns and bounce back, so a shutdown is a buying opportunity.* It surfaces in
  countless explainers — e.g. CNBC, *"Here's how the stock market performs during
  government shutdowns"* (recurring); LPL Financial, *"What Past Government Shutdowns Have
  Meant for Markets"*; Reuters/AP wrap-ups noting the S&P 500 "rose during most past
  shutdowns." The steelmanned hypothesis is testable: the forward total return after a
  federal funding-gap shutdown start is positive *and* larger than you'd earn over the
  same horizon on a random date.

## The shutdown record (the hardcoded event table)

- **Congressional Research Service**, *Shutdown of the Federal Government: Causes,
  Processes, and Effects* (RL34680) — the authoritative catalogue of federal funding-gap
  shutdowns, durations, and which lapses actually furloughed workers vs. brief technical
  gaps. Our table keeps only the furloughing shutdowns in the SPY era (post-1993).
- **U.S. Government Accountability Office** and the **Office of Management and Budget**
  shutdown reports — duration and scope of the 2013 and 2018-19 shutdowns (the longest in
  history, 35 days).
- **Wikipedia**, *Government shutdowns in the United States* — a convenient cross-check of
  start/end dates; reconciled against the CRS list. The brief February-2018 overnight
  lapse (no furlough) is deliberately excluded; the exclusion is stated in `data.py`.

## Why the naive event-study t-stat misleads here

- **Event studies and the right benchmark.** MacKinlay (1997), *Event Studies in
  Economics and Finance* (Journal of Economic Literature) — the canonical reference:
  abnormal returns must be measured against a normal-return benchmark, not against zero.
  A post-event drift that merely matches the market's unconditional drift is *not*
  abnormal. Our synthetic event-null (the same horizon around random dates) is exactly
  that benchmark.
- **Small-sample inference.** With 5 events, a *t*-statistic is dominated by sampling
  noise and a single outlier (here, the 2018-12-24 +12.46%). Multiple-horizon scanning
  (H = 5/10/20/40/60) is itself a soft multiple-testing problem — Harvey, Liu & Zhu
  (2016), *…and the Cross-Section of Expected Returns* (Review of Financial Studies).
- **Equity drift dominates long windows.** Over 40–60 sessions the unconditional SPY
  return is large and positive, so *any* "buy and hold for two months" rule looks good.
  This is the classic confound that makes shutdown (and crisis, and panic) "buy the dip"
  claims look stronger than they are.

## The win-rate / survivorship trap

- **Survivorship of memorable rebounds.** The folklore is built from the shutdowns people
  remember bouncing (2013, 2018-19) and quietly forgets the flat or negative ones
  (1995-96, the −3.99% January-2018 window). An 80% win-rate is 4 of 5 trades — a sample
  far too small to distinguish skill from the base rate of "stocks go up most months."
- **Win-rate vs expectancy.** The desk has dissected the high-win-rate illusion
  repeatedly ([Study 72 — Loaded-Dice](../../72-loaded-dice/),
  [Study 301 — Triple-RSI](../../301-triple-rsi/)). Here it takes the simplest form: a
  high hit-rate on a tiny sample, with the mean indistinguishable from drift.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.hac_tstat`](../government_shutdown/strategy.py).
- **Circular block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap*
  (JASA); Künsch (1989), block bootstrap — [`strategy.block_bootstrap_ci`].
- **Permutation / placebo testing.** Fisher (1935), *The Design of Experiments* — the
  randomisation test underpinning [`strategy.permutation_pvalue`], here used to compare
  shutdown returns against the random-date distribution.

## Data sources used here

- **Yahoo! Finance daily total-return SPY** (via the shared
  `_cache/SPY_total_return.parquet`, `yfinance auto_adjust=True` ⇒ dividends + splits
  folded in ⇒ total return), 1993-01-29 onward. As-of 2026-05-31 with a content
  fingerprint (see [`docs/results.md`](results.md)). The offline reproducible core and
  test-suite run on the deterministic [`data.synthetic_daily`](../government_shutdown/data.py)
  generator, never the network.

## Related desk studies

- **[Study 287 — Easter-Effect](../../287-easter-effect/)** and the pre-holiday family —
  other calendar/event windows where a real-looking drift turns out fragile once raced
  against the right baseline.
- **[Study 301 — Triple-RSI](../../301-triple-rsi/)** and **[Study 72 — Loaded-Dice](../../72-loaded-dice/)**
  — where the high-win-rate illusion is dissected at length.
- The macro-event family (FOMC cycle, October effect) — same trap: an event everyone
  watches, a drift that's mostly just the market doing what it always does.
