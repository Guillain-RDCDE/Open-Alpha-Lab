# References & literature map — Study 13 (Crimson-Hour)

## The claim under test

- **The edgeful newsletter / dashboard.** *"The opening candle closed red on ES & the IB high
  got rejected. 88% of the time, the day closes red"* — edgeful (@edgeful), June 2026.
  <https://www.edgeful.com>. The steelman: on a 1-hour NY-session timeframe, when the
  **09:30→10:30 ET opening candle (OC) closes red** *and* the **initial-balance (IB) high prints
  before the IB low** ("IB-high rejection", so the low is "likely to break first"), the full RTH
  session (09:30→16:00) closes red **22/25 = 88.0%** on ES and **28/31 = 90.3%** on NQ over a
  6-month window (12/04/25→06/04/26, 128 sessions/ticker), against baselines of 46.1% and 44.5%.
  The dashboard was built "in 5 minutes, one prompt, no code" against the edgeful API by combining
  the IB-rejection report with the opening-candle report. The post itself frames the output as
  **"bias, not a guaranteed trade"** and closes on a free-resource / sign-up call to action (the
  "10 dashboards" kit) — the funnel this study keeps in view. We test the gap between *"88% on this
  sample"* and *"the first hour predicts the close."*

- **The tell in the post.** One member example (frank's pre-NFP IB-rejection dashboard) returns
  **64% over 11 sessions ≈ the all-Thursday baseline** — i.e. the same machine that prints 88%
  also prints non-results, but only the winners become headlines. That is the multiple-comparisons
  structure we operationalise.

## Why the steelman is *almost* right — the real effect underneath

- **Intraday return persistence / momentum.** Daily index returns have negligible serial
  dependence (weak-form efficiency, Fama 1970, *Efficient Capital Markets*), but *intraday* there
  is a small, documented first-half-hour → rest-of-day relationship. Heston, Korajczyk &
  Sadka (2010), *Intraday Patterns in the Cross-section of Stock Returns* (Journal of Finance),
  document periodic intraday return continuation. The genuine, modest part of the edgeful claim is
  this continuation — which this study isolates as P(rest-of-day red | OC-red) and finds at only
  ~+6 pp over baseline, versus the ~+25 pp *headline* lift that is mostly a mechanical head-start.
- **Opening range / initial balance.** The IB and opening-range concepts come from J. Peter
  Steidlmayer's Market Profile; "opening-range breakout" trading (e.g. Tony Crabel, 1990,
  *Day Trading With Short Term Price Patterns and Opening Range Breakout*) is the lineage of the
  IB-rejection signal. We test whether the ordering of the IB high vs low adds anything *beyond*
  the opening candle's sign — and find it does not.

## Method lineage (the desk's shared engine)

- **Small-sample binomial inference.** Wilson (1927) score interval, *Probable Inference, the Law
  of Succession, and Statistical Inference* (JASA) — the interval that stays honest at n = 25 and
  near 0/100%, where the textbook normal interval fails. Implemented in
  [`decompose.wilson_ci`](../crimson_hour/decompose.py).
- **Beta-binomial posterior.** A Beta(1,1) prior on the true conditional rate, so "22 of 25" reads
  as a posterior (mean ~85%, wide credible interval) rather than a point — the honest read on a
  headline percentage from a tiny sample. [`decompose.beta_binomial`](../crimson_hour/decompose.py).
- **Data-snooping / the garden of forking paths.** White (2000), *A Reality Check for Data
  Snooping* (Econometrica); Gelman & Loken (2014), *The Garden of Forking Paths*. The "combine
  reports until one hits 88%" workflow is multiple comparisons by another name; our
  [`decompose.mining_inflation`](../crimson_hour/decompose.py) Monte-Carlos the *best* of a bank of
  small-sample confluences and shows a true ~70% edge routinely *presents* as 85–92%.
- **Reproducibility stamp.** [`quantlab/repro.py`](../../../quantlab/repro.py) — the as-of freeze
  and content fingerprint every headline run carries.

## Data sources used here

- **Yahoo! Finance intraday bars** (via `yfinance`). ES=F / NQ=F at **5-minute** fidelity (~60-day
  cap, fine enough to order the IB high vs low) for the faithful confluence; SPY / QQQ at
  **1-hour** fidelity (bars aligned to the 09:30 RTH open, ~730-day reach) for the high-power
  opening-candle leg. SPY/QQQ are the cash proxies for ES/NQ — and edgeful's own members
  (mario's QQQ/SPY scanner) built the same dashboard on them. The intraday window is a *rolling*
  span ending ~now, so every headline is pinned with `as_of` and stamped with a fingerprint.
- **The edgeful API itself is not used.** It would return edgeful's *aggregated* stat, not the raw
  bars the desk's offline core requires; reproducing from public OHLC keeps the study independent
  and rerunnable.

## Related desk studies

- **Study 10 — Markov-Mint** and **Study 12 — Paper-Prophet**: the other "viral, AI-built,
  unusually concrete" quant claims taken apart — there an ARIMA/Markov pipeline; here an
  AI-built dashboard. Same lesson: a real but modest effect, oversold by a layer bolted on top.
- **Study 01 — Overnight-Anomaly**: the night/day return decomposition — the original "normalise
  before you marvel, and separate the mechanical from the forecast" move that this study reuses.
