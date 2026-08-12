# References & literature map — Study 850 (Airline Operational Meltdown)

## The claim under test

- **The folklore.** Every time a carrier melts down operationally — Southwest stranding
  a million travellers over Christmas 2022, Delta grounded worldwide by a datacenter
  outage in 2016 and by the CrowdStrike bug in 2024, United dragging a paying passenger
  off a plane on video in 2017, Boeing's MAX grounded twice — financial commentary reaches
  for the same idea: *the reputational damage will dent the stock*. Customers defect,
  regulators pile on, the brand is bruised, and the equity should carry a lasting mark,
  not just a one-day wobble. This study asks whether that shows up on the tape.
- **The academic anchor.** Event studies of corporate operational disasters and product
  crises (Barber & Darrough 1996 on auto recalls; Chen, Ganesan & Liu 2009 on product
  recalls; Karpoff, Lott & Wehrly 2005 on the reputational vs. legal penalties of
  corporate misconduct) generally find that the **abnormal return concentrates in a
  short window around the announcement** and that the *reputational* component (lost
  future business) is real but small relative to direct **fundamental** costs (recall
  expense, litigation, lost sales). The transferable prior: a shock with genuine
  cash-flow consequences (a grounded product line) moves the stock; a shock that is
  mostly bad press, with the operation restored in days, tends to fade.
- **What "steelmanned" means here.** We give the claim its best shot: we measure not
  just the event-day reaction but the **one-month drift** afterward (the "reputation
  sticks" half of the story), on the single **implicated** ticker (the carrier whose
  brand actually took the hit), with a proper market model that strips out the market's
  own move.

## What we measure, and the honesty rails

- **Market-model event study** (MacKinlay 1997, *Journal of Economic Literature*,
  "Event Studies in Economics and Finance"; Brown & Warner 1985, *Journal of Financial
  Economics*). For each event the "normal" return is `alpha + beta·r_SPY`, with
  `alpha, beta` estimated by OLS over a **120-session estimation window ending 10
  sessions before the event** — so the parameters are not contaminated by the shock. The
  abnormal return is the residual; the CAR is its sum over the event window.
- **Four horizons**, each summarised to one number per event and one-sample *t*-tested
  across the (independent, non-overlapping) events: the event day (offset 0), the event
  week `[0..+4]`, the event-plus-month `[0..+21]`, and the **pure post-event month
  drift** `[+1..+21]` — the last is the direct test of "reputation sticks".
- **Same-ticker random-date permutation placebo.** Each event keeps its own ticker (its
  beta, its idiosyncratic volatility) but is handed a **random pseudo-event date**; over
  5,000 such random calendars we ask how often the mean CAR is as negative as observed.
  A real shock must sit in the left tail; sitting in the bulk means a random calendar of
  the same nine names produces the number anyway.
- **The down-hit rate carries a Wilson (1927) interval**; a **Newey-West** HAC *t* is
  reported as a cross-check but is explicitly **not** the primary — at n = 9 independent
  events the HAC correction is unstable and the plain one-sample *t* is the honest
  statistic.
- **Robustness is the whole story.** The aggregate significance is decomposed
  (airlines-only vs. Boeing-only), leave-one-out'd, and split by sub-era — because the
  headline turns out to hinge on the two Boeing MAX groundings, which are **fundamental**
  (product-line grounding) rather than **reputational** shocks. Naming that is the point
  of the study, not a footnote.
- **Low N, stated up front.** Ten curated meltdowns (nine with price coverage) is a tiny
  sample; the power to detect a small reputational effect is correspondingly low, and the
  default expectation is **None/Weak** unless the CAR is both large and robust. It is not.

## Why the short timer is graded separately

- The tradable overlay — short the implicated stock at the meltdown close, hold a few
  sessions, cover — is a **falsification exercise**: if the reputational-shock claim were
  real and tradable, the short should pay net of costs and borrow. It is positive
  in-sample but never significant and entirely Boeing-concentrated (airlines-only ≈ 0), so
  it is stamped **Fragile**. Costs are 2 × one-way × NAV per round trip plus a 300 bps/yr
  borrow on the short leg; the snap-to-first-session entry is the single documented
  execution lag (see `docs/results.md`). Shorting a stock into a well-publicised meltdown
  also means paying up for hard-to-borrow at exactly the moment the trade is crowded — a
  real-world cost beyond the modelled borrow.

## Data sources

- **SPY** and **LUV / DAL / UAL / AAL / BA** daily total-return closes
  (`auto_adjust=True`) — yfinance (no key), cached under `_cache/` (`amd_spy.csv`,
  `amd_luv.csv`, …), 2014-01-02 → 2026-06-30. **SAVE (Spirit)** is deliberately absent:
  it is fully delisted (Chapter 11 Nov-2024, shares cancelled 2025) and returns no
  history, so the Spirit-2021 meltdown drops out of the real-tape test.
- **10 hardcoded operational meltdowns, 2016 → 2024**, in
  [`airline_meltdown/data.py`](../airline_meltdown/data.py) — each with the implicated
  ticker and a one-line public-record source note referencing the relevant company 8-K,
  the DOT/FAA/NTSB action, and contemporary AP/Reuters/CNBC coverage. No free,
  machine-readable "operational-meltdown index" exists (unlike an FOMC calendar), so this
  is a hand-built calendar of the front-page collapses, cross-referenced against public
  reporting.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [707-plane-crash-effect](../../707-plane-crash-effect/) — major air **crashes** and
  their effect on the broad market and on a 4-carrier airline basket via a
  *sentiment/mood* channel. This study is different on two axes: the trigger is a
  self-inflicted **operational** failure (grounding / cancellation collapse / IT outage /
  PR crisis), not a fatal accident, and the outcome is the **single implicated carrier's
  own stock**, not market-wide mood. (One deliberate near-boundary: the 2019 MAX grounding
  followed a crash, but here it is measured as Boeing's product-line grounding, not as a
  market sentiment shock; 707 excludes it from its market test.)
- [554-airline-bookings](../../554-airline-bookings/) — an **alt-data / demand** signal
  (airline bookings as a fundamental read on the sector), not a discrete reputational
  **event** around a public failure.
- [313-geopolitical-shock](../../313-geopolitical-shock/) — wars, invasions and terror
  attacks moving the **broad market** through a geopolitical/sentiment channel; a
  different trigger, a market-wide (not single-name) outcome, and no operational-failure
  content.

None of the siblings test **what a very public operational meltdown does to the
implicated airline's own stock** — that is this study's own axis.
