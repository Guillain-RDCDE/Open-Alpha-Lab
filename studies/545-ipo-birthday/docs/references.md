# References & literature map — Study 545 (IPO-Birthday)

## The claim

- **Behavioural folklore / attention effect.** The 'birthday effect' for listed companies: the
  anniversary of a firm's IPO each year draws press retrospectives ("one year on the market"),
  analyst calendar notes and investor attention, so the stock should show a small positive abnormal
  return in a window around the anniversary date. This is the calendar/attention cousin of the
  personal-birthday and holiday anomalies below; there is no single canonical academic paper
  establishing an IPO-anniversary return, which is itself informative — the claim lives mostly in
  retail commentary.

## The attention / calendar literature it borrows from

- **Barber & Odean (2008)**, *"All That Glitters: The Effect of Attention and News on the Buying
  Behavior of Individual and Institutional Investors."* *Review of Financial Studies* 21(2). The
  foundational attention-drives-buying result the 'birthday effect' implicitly assumes.
- **Da, Engelberg & Gao (2011)**, *"In Search of Attention."* *Journal of Finance* 66(5). Search /
  attention proxies predict short-run price pressure — the mechanism a fixed calendar anniversary
  would have to exploit.
- **Kliger & Qadan (2018 and related)**, work on **calendar/holiday and birthday-type effects**
  (e.g. Rosh Hashanah–Yom Kippur, "sell on Rosh Hashanah"): documented calendar anomalies are small,
  fragile and often vanish out of sample — the prior we bring to an IPO-anniversary claim.
- **Bouman & Jacobsen (2002)**, *"The Halloween Indicator, 'Sell in May and Go Away.'"* *American
  Economic Review* 92(5). The canonical calendar seasonal — a reminder that even the most-cited
  calendar effects are contested and cost-fragile.

## Shared method — the event study

- **Fama, Fisher, Jensen & Roll (1969)**, *"The Adjustment of Stock Prices to New Information."*
  *International Economic Review* 10(1). The origin of the **cumulative-abnormal-return event
  study**.
- **MacKinlay (1997)**, *"Event Studies in Economics and Finance."* *Journal of Economic
  Literature* 35(1). The standard reference for CAR windows, the market model and abnormal-return
  aggregation this study follows (we use the beta-1 *market-adjusted* variant, appropriate for a
  short symmetric window on a small basket).
- **Label-shuffle / random-anchor permutation testing** (Fisher 1935; Good 2005) — the placebo
  null: re-run the identical window machinery on random, non-anniversary calendar dates and read the
  observed CAR against that distribution.
- **Welch / one-sample *t*** — the inference on the mean event CAR.

## Neighbours on this bench (the dedup map)

- **[Study 219 — IPO-Pop](../../219-ipo-pop/)** — the *first-day* IPO pop and long-run
  underperformance (Ritter 1991). Study 545 is a different object entirely: a **recurring annual
  calendar/attention** effect on the *anniversary*, tested as an event study, not the launch-day
  pop or multi-year drift.
- **[Study 265 — IPO-Volume](../../265-ipo-volume/)** — IPO *issuance* as a market-timing signal;
  again about the IPO market, not an anniversary return on individual names.
- **Calendar cousins** — **[Study 89 — Turn-of-the-Month](../../89-turn-of-the-month/)**,
  **[Study 95 — Holiday-Cheer](../../95-holiday-cheer/)**, **[Study 544 —
  Oyster-R-Months](../../544-oyster-r-months/)**: fixed-calendar seasonals. Study 545 shares the
  method (a calendar anchor + placebo) but anchors on each firm's *own* IPO date rather than a
  market-wide calendar day.

## House methodology

- [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a real-tape *t* ≥ 2 plus a
  placebo null and seed-robustness), the explicit survivorship caveat, one documented execution
  convention, and costs one-way × NAV.
