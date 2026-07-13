# References & literature map — Study 749 (Layoff-Drift)

## The claim under test

- **The folklore ("restructuring pop").** Market commentary routinely reads a mass-layoff
  announcement as *bullish*: cutting headcount signals cost discipline, protects margins,
  and pleases activists — "the stock pops on the restructuring." The 2022–2024 tech
  "efficiency wave" gave this a second life, with a *drift* twist: the stock is said to
  keep climbing as the leaner cost base flows through to earnings (a PEAD-style
  continuation). The bear framing is the mirror image — layoffs signal distress, so the
  stock sags. The believers' one-line question — *"does a mass-layoff announcement pop,
  then drift up?"* — presumes a clean, tradable directional answer. We test whether the
  **abnormal** return around the announcement is a reliable pop, a reliable drift, or too
  noisy and too selection-ridden to trade.
- **What "the market reacts to layoffs" actually means.** The academic question is narrow
  and testable: around a dated layoff announcement, is the **cumulative abnormal return**
  (CAR) — the stock's return net of a market-model benchmark — different from zero on a
  short window and on a longer post-event window?

## What the layoff literature actually finds

- **The announcement reaction is small, mixed, and *not* a clean pop.** Worrell, Davidson
  & Sharma (1991), *Layoff announcements and stockholder wealth* (Academy of Management
  Journal) — average announcement reactions are negative-to-neutral, not the bullish pop
  of folklore; permanent, performance-driven layoffs read worse than reactive ones.
- **The reaction drifts toward zero / mildly negative and is context-dependent.** Farber &
  Hallock (2009), *The Changing Relationship Between Job Loss Announcements and Stock
  Prices: 1970–1999* (Labour Economics) — the market's reaction to layoff announcements has
  become *less* negative over decades but is centered near zero, with the sign driven by
  the *reason* (cost-cutting vs demand shortfall) and by prior expectations. Chen, Mehrotra,
  Sivakumar & Yu (2001) and Nixon, Hitt, Lee & Jeong (2004) survey the wealth effects: no
  robust, sign-stable "layoff = buy" result exists. This is exactly the null our
  short-window pop test confirms.
- **Post-restructuring performance is a survivorship trap.** The firms whose layoffs
  "worked" (margins recovered, stock rallied) are over-represented in any ex-post sample of
  *survivors*; the distressed layoff-announcers that delisted are missing. Any positive
  post-layoff drift measured on a surviving-name panel is therefore an **upper bound**
  (cf. the survivorship-bias literature: Brown, Goetzmann, Ibbotson & Ross, 1992,
  *Survivorship Bias in Performance Studies*, Review of Financial Studies).

## The event-study + PEAD method (the shared engine)

- **Market model / abnormal returns.** Fama, Fisher, Jensen & Roll (1969), *The Adjustment
  of Stock Prices to New Information* (Int. Econ. Review) — the original event study. Brown
  & Warner (1985), *Using daily stock returns: The case of event studies* (J. Financial
  Economics) — the canonical daily-data methodology we follow (estimate `r = α + β·r_mkt`
  on a clean pre-event window, cumulate AR over the event window). MacKinlay (1997),
  *Event Studies in Economics and Finance* (J. Economic Literature) — the textbook synthesis.
- **Post-event drift (PEAD).** Ball & Brown (1968); Bernard & Thomas (1989), *Post-
  Earnings-Announcement Drift: Delayed Price Response or Risk Premium?* (J. Accounting
  Research) — the template for a slow, autocorrelated post-announcement drift, and the
  reason the *drift* leg is tested with a **HAC** t-stat on the pooled daily abnormal series.

## The statistics — why ~two dozen events barely settle it

- **Small-sample inference / power.** With `k ≈ 28` heterogeneous events, the standard
  error of a mean CAR is large and a few outliers dominate. We test each leg's mean against
  zero with a **Welch t** (Welch, 1947), the drift's daily series with a **Newey-West HAC
  t** (Newey & West, 1987, *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix*, Econometrica), and — because `k` is small
  and CARs are heavy-tailed — a **placebo / randomization null** (random non-event
  (ticker, date) windows on the same names; Fisher's randomization logic; Efron &
  Tibshirani, *An Introduction to the Bootstrap*, 1993).
- **Fragility / selection.** A drift that dies when three names are removed, or whose
  bootstrap CI dips below `t = 2`, is fragile by the desk's inference bar (Harvey, Liu &
  Zhu, 2016, *…and the Cross-Section of Expected Returns*; Bailey & López de Prado, 2014,
  *The Deflated Sharpe Ratio* — selection and multiple-window cautions).

## Method lineage (the desk's shared engine)

- **Market-model CAR, pop + drift legs.** [`strategy.event_abnormal`](../layoff_drift/strategy.py)
  and [`strategy.car_panel`](../layoff_drift/strategy.py) — market-model abnormal returns
  cumulated over a short pop window and a long drift window.
- **Welch t, HAC t, placebo p.** [`strategy.welch_t`](../layoff_drift/strategy.py),
  [`strategy.hac_t`](../layoff_drift/strategy.py) (Newey-West on the pooled daily drift),
  and [`strategy.placebo_pvalue`](../layoff_drift/strategy.py) — the Signal-axis tests.
- **Execution lag + costs.** `event_abnormal(..., lag=1)` models a one-day execution delay;
  [`strategy.net_of_costs`](../layoff_drift/strategy.py) applies a one-way round-trip.
- **Deterministic synthetic control.**
  [`data.synthetic_events`](../layoff_drift/data.py) plants known pop/drift edges; the
  offline core runs with no network. The control confirms the engine recovers a planted
  drift **and** that ~two dozen events cannot reach significance unless the drift is large.

## Data sources used here

- **Hardcoded event table** (`layoff_drift.data.LAYOFF_EVENTS`): ~28 notable large-cap
  layoff announcements (ticker, date, approximate cut), compiled from company press
  releases and contemporaneous WSJ / Reuters / Bloomberg / FT / Layoffs.fyi coverage.
  There is no free, survivorship-clean database of mass-layoff dates; the dated table is
  the transparent stand-in. **Survivorship** (only long-listed survivors are priceable) is
  named on the Signal axis — it biases the drift *up*.
- **yfinance** daily adjusted (total-return) closes for each event ticker + SPY, cached
  under `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and
  reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 391 — CEO-Turnover](../391-ceo-turnover/)**: the sibling rare-catalyst event
  study — a real day-one repricing you can't trade, weak over any holdable window. Same
  small-sample event-study pathology.
- **[Study 389 — Name-Change-Effect](../389-name-change-effect/)**: the pop-then-fade
  legend on theme-chasing rebrands — another "the pop is remembered because the losers
  delisted" survivorship story.
