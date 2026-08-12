# References & literature map — Study 857 (SG&A Cost Stickiness)

## The claim under test

- **The thesis (Anderson, Banker & Janakiraman 2003).** In *"Are Selling, General, and
  Administrative Costs 'Sticky'?"* (**Journal of Accounting Research** 41(1), 47-63) ABJ show
  that SG&A costs do **not** move symmetrically with activity: across a large Compustat panel
  they rise about **0.55%** for a 1% increase in sales but fall only about **0.35%** for a 1%
  *decrease* in sales. Managers, facing adjustment costs and reluctant to scrap sales capacity
  or a marketing programme they may need again, **delay cutting** discretionary overhead into a
  downturn — costs are "sticky" on the way down. Their identifying regression, on annual data,
  is

  ```
  Δlog(SG&A_t) = β0 + β1·Δlog(Sales_t) + β2·(Decrease_t · Δlog(Sales_t)) + ε_t
  ```

  where `Decrease_t = 1` when sales fell year-over-year. `β1` is the response to a sales
  increase, `β1 + β2` the response to a decrease, and **`β2 < 0` is the signature of
  stickiness**. We define a firm's **stickiness = −β2** (bigger ⇒ stickier).
- **The trading hypothesis we test.** The economic reading of stickiness is *weaker operating
  discipline* — a sticky firm carries slack overhead into declines. If markets under-appreciate
  that, **sticky-cost firms should under-earn**: lower forward returns and weaker forward
  profitability than lean, disciplined peers. This connects cost stickiness to the
  cost-management and earnings-quality literature (Banker & Chen 2006 on cost behaviour and
  earnings prediction; Weiss 2010, *The Accounting Review*, "Cost behavior and analysts'
  earnings forecasts", who shows sticky-cost firms have less accurate, more pessimistically-
  surprising earnings). Whether that maps into a *tradeable forward-return spread* is the open
  question, and the desk's usual answer is "probably not once you demand a robust *t*."
- **The academic anchor for the return claim.** This is a specific instance of the broad
  **fundamentals-to-returns anomaly** family (Sloan 1996 on accruals; Novy-Marx 2013 on
  gross profitability; Ball, Gerakos, Linnainmaa & Nikolaev 2015 on operating profitability).
  Cost stickiness is a *cost-side operating-quality* characteristic; the natural null is that a
  publicly-computable regression coefficient from years-old filings is already in prices.

## What we measure, and the honesty rails

- **Signal, point-in-time.** Per firm we build **quarterly year-over-year log changes** in SG&A
  and in sales and a sales-decrease dummy, then at **each 10-Q/10-K filing date** re-estimate
  the ABJ regression on an **expanding window of only the observations already public** (never
  the full-sample coefficient — that would leak the future). The event carries
  `stickiness = −β2` and the mirror trading signal `disc = β2` (higher ⇒ leaner). *Design
  choice:* ABJ estimate on annual data; we use quarterly YoY changes so a point-in-time
  expanding window has enough observations to identify β2 — the same asymmetry, at a frequency
  that supports a tradeable, evolving signal (as in the quarterly-stickiness follow-up
  literature).
- **Identification is a first-class constraint.** β2 is only identified once a name has lived
  through several YoY sales **declines** in its public window. We require `MIN_OBS = 20`
  observations and `MIN_DEC = 4` decline quarters before trusting an estimate; steady growers
  drop out or arrive late. This is honest, and it tilts the estimable panel toward cyclicals.
- **Primary test — calendar-time long-short (Newey-West).** Each month-end, rank the names that
  carry a fresh estimate into **terciles** on `disc` (the panel is too thin for quintiles),
  long the top (leanest) / short the bottom (stickiest) equal-weight, earn the **next** month's
  return (one execution lag). The decisive statistic is the **Newey-West (HAC, Bartlett) t** of
  the monthly long-short series — the autocorrelation-robust bar `REAL` is written against
  (METHODOLOGY → *The inference bar*). A one-sample t and a monthly hit-rate accompany it.
- **Cross-check — pooled event drift.** Bucket all (ticker, filing) events by `disc`, measure
  top-minus-bottom forward drift over ≈1m/1q/2q horizons, with a one-sample t and a
  **label-shuffle placebo** (permute signals, re-form random terciles), plus the tercile
  monotonicity picture.
- **Third axis — does stickiness predict weaker future profitability?** A pooled OLS of the
  change in trailing-four-quarter ROA over the next ~year on the stickiness measure. The claim
  predicts a **negative** slope (sticky firms' ROA falls further). This is the *mechanism*
  check: weaker operating discipline can show up in the fundamentals even if the stock does not
  move. The pooled t ignores firm/quarter clustering, so we read the magnitude, not the literal
  t.
- **Costs & borrow.** The tradability timer charges one-way cost × NAV × monthly turnover on
  **both** legs and makes the short leg pay an annualised borrow — the standard desk long-short
  friction model. One documented execution lag throughout.
- **Coverage is a caveat, not a footnote.** EDGAR XBRL history begins ~2009; the expanding-
  window estimate only matures ~4-5 years later, so the point-in-time signal is available from
  ≈2015, and the cross-section is dominated by names that actually decline (semis, industrials,
  autos, hardware). Terciles on a thin cross-section are noisy by construction.

## Survivorship — named on the Signal axis

The basket is **current survivors** (all still listed): a fixed roster of ~33 large filers that
report a current `SellingGeneralAndAdministrativeExpense` line with deep history. It cannot
include firms that were acquired or failed — and, pointedly, the firms whose *lack of cost
discipline* actually killed them are exactly the ones missing. That truncation runs **against**
finding that sticky firms under-earn (the worst sticky outcomes are censored), so a null here is
conservative and a positive would have to survive that bias. We reason about the direction
explicitly rather than claiming it away, and never cite the survivor panel to certify magnitude.

## Data sources

- **SG&A, revenue, net income, total assets** — SEC EDGAR XBRL `companyconcept` API
  (`data.sec.gov`): `SellingGeneralAndAdministrativeExpense`, `Revenues` /
  `RevenueFromContractWithCustomer…`, `NetIncomeLoss`, `Assets`; 10-Q/10-K facts, de-duplicated
  on period end (earliest filing kept), keeping the filing date so the signal is strictly
  point-in-time. Cached under `_cache/sga_events.csv`.
- **Daily adjusted closes** — yfinance (no key), cached under `_cache/sga_prices.csv`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [524-operating-leverage](../524-operating-leverage/) — the **cost-structure / fixed-vs-
  variable** angle: how a high fixed-cost base *amplifies* earnings swings. That is about the
  *magnitude* of the cost-to-sales elasticity; this study is about its **asymmetry** (up vs
  down), the ABJ β2, which operating leverage does not measure.
- [200-roe-quality](../200-roe-quality/) — ranks on the **level/quality of ROE**. Cost
  stickiness is a *behavioural cost-dynamics* characteristic estimated from a time-series
  regression per firm, not a profitability level.
- [122-gross-profitability](../122-gross-profitability/) — Novy-Marx gross profits / assets, a
  *level* profitability signal on the top of the income statement. This study is a
  *below-the-line SG&A cost-behaviour* signal, orthogonal in construction.
- [749-layoff-drift](../749-layoff-drift/) — the drift after an **announced layoff** (a discrete
  cost-cutting *event*). Stickiness is the *opposite* posture — the firms that *don't* cut — and
  is a slow estimated characteristic, not an event study.

None of the siblings estimate the **asymmetric SG&A-to-sales response (ABJ cost stickiness)** —
this study's own axis.
