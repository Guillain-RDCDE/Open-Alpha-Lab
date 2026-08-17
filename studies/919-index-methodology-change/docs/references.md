# References & literature map — Study 919 (Methodology Shock)

## The claim under test

- **The index-flow thesis.** An index is a rulebook, and every wrapper that tracks it is
  contractually obliged to hold whatever the rulebook says. When the *rules themselves*
  change — a special rebalance to cap concentration, a shift to float-adjusted weights, a
  new eligibility filter, a change to how constituents migrate between indexes — trillions
  of dollars of passive money must trade a large basket on a known date. The folk claim is
  that this creates a predictable, front-runnable footprint: buy the affected wrapper
  against an unaffected sibling between the announcement and the effective date.
- **The steelman.** The mechanism is real at the *constituent* level and well documented
  (see the demand-curve literature below). Study 919 asks the narrower, tradable question
  a price-taker can actually act on: does the footprint survive **at the wrapper level**,
  where the reweighting is diluted across hundreds of names, once you hedge the market
  exposure and pay to cross four spreads?

## The events tested (the hardcoded PROXY calendar)

The event list in [`methodology_shock/data.py`](../methodology_shock/data.py) is
hand-assembled from public index-provider announcements. It is the study's one non-tape
input and is labelled a PROXY/ASSUMPTION throughout; each row carries a `date_confidence`
stamp and the analysis sweeps windows from ±1 to ±10 sessions precisely so that a
few-session date error cannot drive the result.

- **Nasdaq-100 special rebalances (2011, 2023).** Nasdaq's methodology caps the aggregate
  weight of the largest constituents; twice the cap was breached badly enough to force an
  off-cycle special rebalance — in 2011 (Apple's weight cut from roughly 20.5% to 12.3%)
  and in 2023 (the "Magnificent Seven" concentration fix). Treated wrapper: **QQQ**;
  sibling: **SPY**.
  - *Provenance, 2011 row (corrected in audit).* NASDAQ OMX announced the special
    rebalance on **2011-04-05**, effective **2011-05-02**; it was reported the same day by
    CNNMoney ("Apple's influence on Nasdaq-100 index slashed by 40%", 2011-04-05), CNBC
    and Nasdaq's own newsroom. The first build of this calendar carried **2011-03-24** and
    stamped it `exact`. That was wrong. Correcting it flipped the event's
    announcement-leg CAR from −120.1 bps to **+36.3 bps** and the pooled announcement leg
    from −36.4 bps to −14.1 bps. The correction is documented rather than absorbed
    because it is the cleanest available demonstration of why this input is labelled a
    PROXY: in a seven-observation study, one mis-transcribed date is worth 22 bps of
    headline.
- **S&P 500 float adjustment (2004 announcement, 2005 two-phase transition).** S&P moved
  its US indices from total-shares to float-adjusted weights in two steps, a mechanical
  reweighting of the entire index. Treated: **SPY**; sibling: **IWM** (a non-S&P index the
  change did not touch). The two phases share a single announcement, so only the second
  contributes an independent *effective*-date observation.
- **S&P 500 multiple-share-class eligibility (2017 exclusion, 2023 reversal).** S&P first
  barred, then re-admitted, companies with multiple share classes — a change to *who may
  be in the index* rather than to the weights. Treated: **SPY**; sibling: **IWM**. Both
  took effect on announcement, so their announcement and effective legs coincide — which
  means **2 of the 7 observations are shared between the two legs**. The legs are
  reported side by side as a sign check and are never pooled or counted as 14 draws.
- **Russell reconstitution rules (banding; the move to semi-annual reconstitution).**
  FTSE Russell's banding rule damps index migration at the Russell 1000/2000 boundary,
  and the announced shift from annual to semi-annual reconstitution changes the cadence of
  the largest scheduled trade in US equities. Treated: **IWM**; sibling: **MDY** (S&P
  MidCap 400, a comparable small/mid wrapper on a different rulebook). The semi-annual
  change goes live in November 2026, past this study's as-of, so only its announcement leg
  is tested.

## Why the mechanism is real at the constituent level

- **Shleifer (1986), *Do Demand Curves for Stocks Slope Down?*, Journal of Finance** — the
  founding S&P 500 inclusion study: added stocks jump, and the jump is larger when index
  ownership is larger. The canonical evidence that index demand moves prices.
- **Harris & Gurel (1986), *Price and Volume Effects Associated with Changes in the S&P
  500*, Journal of Finance** — the same effect with substantial reversal, the first hint
  that the footprint is liquidity provision rather than a permanent revaluation.
- **Wurgler & Zhuravskaya (2002), *Does Arbitrage Flatten Demand Curves for Stocks?*,
  Journal of Business** — the size of the inclusion effect tracks how hard the stock is to
  arbitrage; it is a limits-to-arbitrage phenomenon, not a free lunch.
- **Chen, Noronha & Singal (2004), *The Price Response to S&P 500 Index Additions and
  Deletions*, Journal of Finance** — the asymmetry between additions and deletions, and
  the investor-awareness channel.
- **Petajisto (2011), *The Index Premium and Its Hidden Cost for Index Funds*, Journal of
  Empirical Finance** — quantifies what the index-rebalancing footprint costs the funds
  that must trade it. This is the *other* side of the trade this study tries to take, and
  it is the reason a wrapper-level effect should be expected to be small: the cost is paid
  at the constituent level and diluted across the basket.
- **Greenwood (2005), *Short- and Long-Term Demand Curves for Stocks*, Journal of
  Financial Economics** — the Nikkei 225 weighting-rule change as a clean natural
  experiment in *methodology*-driven (not membership-driven) flow, the closest published
  analogue to the events tested here.

## Why the wrapper-level version can be empty

- **Dilution.** A special rebalance that halves one constituent's weight moves the index
  by the product of that weight change and the stock's idiosyncratic move — typically tens
  of basis points, not hundreds. The tested design's minimum detectable CAR is ~114 bps.
- **Anticipation.** Index rule changes are announced weeks ahead precisely so that
  trackers can prepare. By the time the rule is public, the constituent-level repricing has
  been arbitraged; nothing is left at the index level.
- **Decay of the inclusion effect.** Bennett, Stulz & Wang (2020), *Does Joining the S&P
  500 Index Hurt Firms?* (NBER 27593), document that the S&P 500 addition premium has
  shrunk toward zero in the 2010s as index arbitrage capital grew. A wrapper-level residual
  of an already-shrinking constituent-level effect is the smallest thing in the chain.
- **Small-sample inference.** Fewer than ten rule changes in three decades makes the
  ordinary *t*-statistic untrustworthy — as this study's `[−5,−1]` window demonstrates
  (naive *t* = −3.43, randomisation *p* = 0.45). MacKinlay (1997), *Event Studies in
  Economics and Finance*, Journal of Economic Literature, is the standard reference for the
  market-model event-study design and its small-sample caveats.

## Related desk studies (dedup)

- **[Study 320 — Russell-Reconstitution](../../320-russell-reconstitution/)**: front-running
  the *annual, scheduled* late-June Russell reshuffle by buying IWM ahead of reconstitution
  Friday. That is the recurring **calendar** event; Study 919 tests the rare, one-off
  changes to the **rulebook itself** (banding, the move to semi-annual recon), which have
  their own announcement dates and no seasonality.
- **[Study 604 — Month-End Rebalancing Flows](../../604-month-end-rebalancing-flows/)** and
  **[Study 836 — Rebalance Timing Luck](../../836-timing-luck/)**: the flow and the
  path-dependence created by *portfolio* rebalancing calendars, not by an index provider
  changing its construction rules.
- **[Study 102 — Free-Rebalance](../../102-free-rebalance/)**: whether rebalancing itself
  adds return. A portfolio-construction question, orthogonal to index methodology.
- **Study 913 (tracking difference) and Study 918 (creation halts)**, its neighbours in the
  same lot, are *wrapper-quality* and *wrapper-plumbing* races. Study 919 is the only one
  that treats the **index rulebook** as the event.

## Method lineage

- **Market-model event study & CAR.** MacKinlay (1997) and Brown & Warner (1985),
  *Using Daily Stock Returns: The Case of Event Studies*, Journal of Financial Economics —
  the estimation-window/event-window separation and the daily-data caveats implemented in
  [`strategy.event_abnormal`](../methodology_shock/strategy.py).
- **Randomisation / placebo inference.** Fisher's permutation logic, in the modern form of
  Bertrand, Duflo & Mullainathan (2004), *How Much Should We Trust Differences-in-
  Differences Estimates?*, QJE — placebo events drawn from the same tape are the honest
  yardstick when the treated sample is tiny.
  [`strategy.placebo_test`](../methodology_shock/strategy.py).
- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite …
  Covariance Matrix*, Econometrica — [`strategy.newey_west_t`](../methodology_shock/strategy.py)
  and [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Circular block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap*, JASA —
  [`strategy.block_bootstrap_ar_ci`](../methodology_shock/strategy.py) and
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).
- **Multiple testing.** Harvey, Liu & Zhu (2016), *… and the Cross-Section of Expected
  Returns*, Review of Financial Studies — the reason the nine-window sweep reports a
  Bonferroni-adjusted *p* alongside the raw one.

## Data sources

- **QQQ** (Nasdaq-100), **SPY** (S&P 500), **IWM** (Russell 2000), **MDY** (S&P MidCap
  400) and **BIL** (1-3M T-bill, the cash leg) — daily **total-return** closes via
  `yfinance` (`auto_adjust=True`), read from the shared desk cache at `studies/_cache`,
  1993 → 2026-06-30. Total return matters here: QQQ's distribution yield is roughly a
  percent below SPY's, so a price-only QQQ-minus-SPY leg would carry a spurious negative
  drift of about the size of the effect being hunted.
- **As-of 2026-06-30.** The partial current month is dropped so the sample never creeps.
- **The event calendar is not a data source.** It is a hardcoded, hand-assembled proxy
  (see above), and the study's robustness apparatus exists mainly to bound what a wrong
  date could do to the answer. The 2011 correction above is a worked example of exactly
  that failure mode being caught.
- **BIL's history is shorter than the event list.** The cash leg starts 2007-05-30, so the
  2004 and April-2007 announcements fall outside it. The deployed-capital race **drops**
  them and says so; it does not map them onto BIL's first session (which an earlier build
  did, manufacturing a −3.8% drawdown in the first ten days of the cash era out of two
  events that predate it).

## The non-tape assumptions, named

| Assumption | Default | Swept over | Where |
|---|---|---|---|
| Event calendar (dates + treated/control pairing) | 8 hand-transcribed rows, 3 `exact` / 5 `approximate` | ±1 to ±10 trading-day window sweep; drop-one jackknife; 2,000-draw placebo | `data.EVENTS` |
| Round-trip cost | 5 bps one-way × NAV | 0 / 1 / 5 / 10 / 25 bps | `strategy.cost_sweep` |
| Borrow on the short leg | 50 bps/yr | 0 / 50 / 100 / 300 bps/yr | `strategy.cost_sweep` |
| Financing on the residual `1 − beta` dollar exposure | 200 bps/yr | 0 / 200 / 500 bps/yr | `strategy.finance_sweep` |

The beta-hedged pair is *not* dollar-neutral — long 1 against short `beta` leaves
`1 − beta` units of NAV as a real net position (net long +0.52 at beta 0.48, net short
−0.23 at beta 1.23). Charging (or crediting) that residual at a bill rate is what makes
the reported net an excess-of-cash return rather than one asserted to be. Only the naive
1×/1× variant finances itself exactly.
