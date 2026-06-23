# References & literature map — Study 391 (CEO-Turnover)

## The claim under test

- **The folklore.** Market commentary frames a CEO firing both ways at once: a *forced*
  exit is sometimes cheered ("the activists won, the underperformer is gone — buy the
  relief rally") and sometimes feared ("leadership chaos, sell the uncertainty"). The
  believers' one-line question — *"Is firing the CEO good news or bad news for the
  stock?"* — presumes a clean, tradable directional answer exists. We test whether the
  *abnormal* return around the announcement is reliably positive, reliably negative, or
  simply too noisy and too rare to call.
- **What "the market reacts to CEO changes" actually means.** The academic question is
  narrow and testable: around a dated CEO-change announcement, is the **cumulative
  abnormal return** (CAR) — the stock's return net of a market-model benchmark —
  different from zero, and does it differ between **forced** and **planned** departures?

## The event-study method (the shared engine)

- **Market model / abnormal returns.** Fama, Fisher, Jensen & Roll (1969), *The
  Adjustment of Stock Prices to New Information* (Int. Econ. Review) — the original
  event study. Brown & Warner (1985), *Using daily stock returns: The case of event
  studies* (J. Financial Economics) — the canonical daily-data methodology we follow:
  estimate `r_stock = α + β·r_mkt` on a clean pre-event window, then cumulate the
  abnormal return AR = r_stock − (α + β·r_mkt) over a short event window. MacKinlay
  (1997), *Event Studies in Economics and Finance* (J. Economic Literature) — the
  textbook synthesis (estimation window, event window, CAR, t-tests).

## What the literature actually finds on CEO turnover

- **Forced turnover follows poor performance.** Coughlan & Schmidt (1985); Warner,
  Watts & Wruck (1988), *Stock prices and top management changes* (JFE); Weisbach
  (1988); Jenter & Kanaan (2015), *CEO Turnover and Relative Performance Evaluation*
  (J. Finance) — forced turnover is *predicted by* prior bad returns (the announcement
  is partly anticipated), which blunts and muddies the announcement-day reaction.
- **The announcement reaction is small, mixed, and context-dependent.** Denis & Denis
  (1995); Huson, Malatesta & Parrino (2004), *Managerial succession and firm
  performance* (JFE) — average announcement returns to management changes are economically
  small and statistically fragile; the sign depends on whether the successor is an
  outsider, whether the firm was already in play, and on prior expectations. There is
  **no robust, sign-stable "fire-the-CEO" trade** in this literature — exactly the null
  our short-window test confirms on a small modern large-cap sample.

## Why ~a dozen events per bucket cannot settle it — the statistics

- **Small-sample inference / power.** With `k ≈ 11–12` events per bucket, the standard
  error of a mean CAR is large; a forced-minus-planned gap of a couple of points cannot
  be distinguished from luck. We test each bucket's mean against zero and the
  forced−planned difference with a **Welch t** (Welch, 1947), and — because `k` is tiny
  and CARs are heavy-tailed — with a **placebo / randomization null**: random non-event
  (ticker, date) windows on the same names (Fisher's randomization logic; Efron &
  Tibshirani, *An Introduction to the Bootstrap*, 1993).
- **Multiple windows / selection.** A single-day [0,0] window can cross t≈2 while the
  holdable [0,+2] window does not — a reminder that an event "reaction" measured on the
  exact print is not the same as a tradable post-announcement drift (Harvey, Liu & Zhu,
  2016, *…and the Cross-Section of Expected Returns*; the look-ahead and window-mining
  cautions of Bailey & López de Prado, 2014, *The Deflated Sharpe Ratio*).

## Method lineage (the desk's shared engine)

- **Market-model CAR + buckets.** [`strategy.event_car`](../ceo_turnover/strategy.py) and
  [`strategy.car_panel`](../ceo_turnover/strategy.py) — market-model abnormal returns
  cumulated over the event window, split by forced/planned.
- **Welch t + placebo p-value.** [`strategy.welch_t`](../ceo_turnover/strategy.py) and
  [`strategy.placebo_car_dist`](../ceo_turnover/strategy.py) — the Signal-axis tests:
  bucket mean vs zero, forced−planned difference, and a random-window null.
- **Execution lag + costs.** `event_car(..., lag=1)` models a one-day execution delay
  (you only learn the news at the close); [`strategy.net_of_costs`](../ceo_turnover/strategy.py)
  applies a one-way round-trip.
- **Deterministic synthetic control.**
  [`data.synthetic_events`](../ceo_turnover/data.py) plants a known forced-bucket CAR
  edge; the offline core runs with no network. The control confirms the engine recovers
  a planted edge **and** that ~a dozen events per bucket cannot reach significance unless
  the planted edge is large.

## Data sources used here

- **Hardcoded event table** (`ceo_turnover.data.CEO_EVENTS`): ~25 notable large-cap CEO
  changes (ticker, announcement date, forced/planned), compiled from company press
  releases and contemporaneous WSJ / Reuters / Bloomberg / FT coverage. True
  CEO-turnover databases (S&P ExecuComp, BoardEx) are not free; the dated, labelled
  table is the transparent stand-in. The forced/planned label is the believers' own
  framing and is subjective at the margin — named on the Signal axis.
- **yfinance** daily adjusted closes for each event ticker + SPY, cached under
  `_cache/`. Two delisted names (TWTR, YHOO) and one pre-IPO event (UBER 2017) drop out
  for lack of price history — a mild **survivorship** tilt named on the Signal axis. All
  headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 249 — Index-Inclusion](../249-index-inclusion/)**: another short-window
  corporate-event study (the S&P inclusion "pop"). There the event is *frequent* and the
  print *measurable but tiny*; here the event is *rare and heterogeneous*, so even the
  sign is unstable — two faces of event-study power.
- **[Study 369 — Earnings-Revision-Momentum](../369-earnings-revision-momentum/)** and
  **[Study 228 — Pre-Earnings-Runup](../228-pre-earnings-runup/)**: other dated-event
  reactions where the question is whether a known catalyst leaves a *capturable* drift.
