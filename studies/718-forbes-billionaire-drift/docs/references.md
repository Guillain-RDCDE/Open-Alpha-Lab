# References & literature map — Study 718 (Forbes-Billionaire-Drift)

## The claim under test

- **The folklore.** Every spring, the [Forbes World's Billionaires
  list](https://www.forbes.com/billionaires/) (published annually since 1987; Forbes,
  Kroll & Dolan, eds.) mints a fresh crop of founders, and market commentary turns it into
  a trade: *"a founder just became a billionaire — the company is on fire — buy the public
  vehicle."* The believers' one-line question — *"buy the newly-minted billionaire's
  stock?"* — presumes a tradable **forward** drift exists after the list is public.
- **What the tradable version actually means.** The academic question is narrow and
  testable: around the dated list-publication day, is the **cumulative abnormal return**
  (CAR) of the founder's vehicle — the stock's return net of a market/factor benchmark —
  reliably positive **after** the list is public (when you could buy), and is that positive
  drift *robust to the benchmark*, or is it just the growth-factor beta these names carry?

## The reverse-causality / selection trap (the heart of this study)

- **Conditioning on a survivor of a run-up.** List membership is defined by wealth ≈
  shares × price, so "newly crossed \$1B this year" conditions on a large trailing price
  run. The pre-list abnormal return is therefore positive *by construction* with no
  forecasting content — a textbook **selection / look-ahead** artifact. The general hazard:
  Brown, Goetzmann, Ibbotson & Ross (1992), *Survivorship Bias in Performance Studies*
  (Review of Financial Studies); and the "picking winners after the fact" caution in
  Harvey, Liu & Zhu (2016), *…and the Cross-Section of Expected Returns* (RFS).
- **Rich-list membership and subsequent returns.** The nearest empirical literature finds
  *no* reliable post-membership outperformance and often the opposite: Kacperczyk, Nosal &
  Sundaresan and the broad "founder/insider wealth" strand document that extreme prior
  run-ups mean-revert; media-salient "hot" stocks underperform (Fang & Peress, 2009,
  *Media Coverage and the Cross-Section of Stock Returns*, J. Finance). A "buy the
  celebrated name" rule is closer to the **attention-driven / lottery-stock** losers of
  Barber & Odean (2008) and Bali, Cakici & Whitelaw (2011, *Maxing Out*, JFE) than to any
  edge.

## The event-study method (the shared engine)

- **Market model / abnormal returns.** Fama, Fisher, Jensen & Roll (1969), *The Adjustment
  of Stock Prices to New Information* (Int. Econ. Review) — the original event study. Brown
  & Warner (1985), *Using daily stock returns: The case of event studies* (JFE) — the
  canonical daily-data methodology: estimate `r_stock = α + β·r_mkt` on a clean pre-event
  window, cumulate AR = r_stock − (α + β·r_mkt) over the event window. MacKinlay (1997),
  *Event Studies in Economics and Finance* (J. Econ. Literature) — the textbook synthesis.
- **Beta instability on young / high-vol stocks.** A short-window market-model β on a
  freshly-IPO'd, high-volatility name is noisy, so a fitted "abnormal return" can diverge
  sharply from a plain excess return (Scholes & Williams, 1977; Dimson, 1979 — thin-trading
  and estimation-window betas). We turn that into a diagnostic: comparing the SPY
  market-model CAR to a QQQ benchmark and to a plain β = 1 excess return is our
  **alpha-vs-beta** test — most of the "drift" is the growth factor the S&P omits.

## The statistics (why ~25 high-vol events can't settle it)

- **Small-sample inference / power.** With `k = 25` ultra-high-vol events, the standard
  error of a mean CAR is large; a double-digit point estimate can still fail `t = 2`. We
  test the mean against zero with a **Welch t** (Welch, 1947) and, because `k` is small and
  CARs are heavy-tailed, with a **placebo / randomization null** — random non-event
  (ticker, date) windows on the same names (Fisher's randomization logic; Efron &
  Tibshirani, *An Introduction to the Bootstrap*, 1993).
- **Multiple windows / benchmarks as researcher DoF.** Which window (pre/announce/post) and
  which benchmark (SPY/QQQ/β = 1) you pick moves the answer — precisely the data-snooping
  and look-ahead cautions of Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*,
  which is why we pre-declare the *holdable, benchmark-robust* post-list window as the only
  one that would count.

## Method lineage (the desk's shared engine)

- **Market-model CAR + windows.**
  [`strategy.event_car`](../forbes_billionaire_drift/strategy.py) /
  [`strategy.car_panel`](../forbes_billionaire_drift/strategy.py) — market-model abnormal
  returns cumulated over pre/announce/post windows, with a swappable benchmark.
- **Alpha-vs-beta decomposition.**
  [`strategy.raw_excess_panel`](../forbes_billionaire_drift/strategy.py) — the plain
  β = 1 excess return that shows the fitted-beta "drift" is an artifact.
- **Welch t + placebo p-value.** [`strategy.welch_t`](../forbes_billionaire_drift/strategy.py)
  and [`strategy.placebo_car_dist`](../forbes_billionaire_drift/strategy.py).
- **Execution lag + costs.** `event_car(..., lag=1)` (enter the day after publication) and
  [`strategy.net_of_costs`](../forbes_billionaire_drift/strategy.py) (one-way round-trip).
- **Deterministic synthetic control.**
  [`data.synthetic_events`](../forbes_billionaire_drift/data.py) plants a known post-list
  drift; the offline core runs with no network. The control confirms the engine recovers a
  planted edge **and** that ~25 ultra-vol events cannot reach significance unless the
  planted edge is enormous.

## Data sources used here

- **Hardcoded event table** (`forbes_billionaire_drift.data.FORBES_EVENTS`): ~27 newly-
  minted-billionaire vehicles (ticker, founder, first-annual-list year, list date),
  compiled from the Forbes World's Billionaires list and company IPO/listing records &
  contemporaneous financial-press coverage. Forbes does not license a machine-readable
  new-entrants feed; the dated, cited table is the transparent stand-in. The "newly-minted"
  call is Forbes' own framing and subjective at the margin — named on the Signal axis.
- **yfinance** daily adjusted closes for each event ticker + **SPY** (broad-market
  benchmark) + **QQQ** (tech-beta cross-check), cached under `_cache/`. Two delisted names
  — **LAZR** (Luminar) and **NKLA** (Nikola) — drop out for lack of a continuous price
  series, a **survivorship** tilt that biases the survivor mean *upward*, named on the
  Signal axis. All headline numbers are pinned in [`docs/results.md`](results.md) and
  reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 391 — CEO-Turnover](../391-ceo-turnover/)**: the same market-model event study on
  a dated corporate catalyst, where the only real move is the un-tradable announcement
  instant — a close cousin of the "real but untradable" run-up and wrong-sign dip here.
- **[Study 389 — Name-Change-Effect](../389-name-change-effect/)**: a selection-on-vivid-
  survivors illusion (Long Blockchain, KodakCoin) — Nikola and Luminar are this study's
  version of the same trap.
