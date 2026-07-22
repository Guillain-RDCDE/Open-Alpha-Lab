# References & literature map — Study 798 (Deferred-Revenue Signal)

## The claim under test

- **The thesis.** For a subscription/SaaS business, **deferred revenue** (a.k.a. *contract
  liabilities* under ASC 606) is cash already billed for services not yet delivered — an
  annual SaaS contract signed today is parked as a liability and recognised into revenue over
  the next four quarters. Because the balance is *bookings that have not yet hit the income
  statement*, a swelling deferred-revenue balance is argued to **lead future sales and, if the
  market is slow to price it, forward stock returns**. The retail/sell-side version: "watch
  deferred revenue growth, it front-runs the revenue print." We test the strong form: rank
  names on **year-over-year growth in current deferred revenue** and measure the forward
  long-short return spread.
- **The academic anchor.** The signal is a specific instance of the **accrual / balance-sheet
  information anomaly**. Sloan (1996, *The Accounting Review*) showed the market misprices the
  accrual vs cash components of earnings; deferred revenue is an unusual, *good-news* accrual
  (a liability that predicts future revenue rather than reversing against it). The specific
  deferred-revenue result is documented in **Prakash & Sinha (2013, "Deferred Revenues and the
  Matching of Revenues and Expenses", *Contemporary Accounting Research*)** and in practitioner
  work on **"billings" and calculated deferred-revenue growth** as a SaaS quality signal. The
  mechanism — order intake leading recognised revenue — is the same one behind classic
  **order-backlog** research (Rajgopal, Shevlin & Venkatachalam 2003, *RAST*), the sibling
  study below.
- **The open question we test.** On a small, honestly-thin panel of names that actually report
  a current deferred-revenue / contract-liability balance on EDGAR, does the growth signal
  (a) **lead future sales** (the accounting claim) and (b) **earn a forward return spread** (the
  market-mispricing claim), once you rank strictly on point-in-time filed values, hold with one
  execution lag, and charge realistic long-short costs plus borrow?

## What we measure, and the honesty rails

- **Signal, point-in-time.** `defrev_yoy` = current-quarter deferred-revenue balance ÷ the same
  quarter a year earlier − 1, known only at the **10-Q/10-K filing date** (`filed`), never the
  period end. A balance-sheet-scaled variant (`(ΔDeferredRevenue)/Assets`) is carried as a
  robustness cut. Concept: `DeferredRevenueCurrent` for older filers, falling back to the
  ASC-606 successor `ContractWithCustomerLiabilityCurrent` post-2018; we take the concept with
  the longer per-name history.
- **Primary test — calendar-time long-short (Newey-West).** Each month-end, rank the names that
  carry a fresh signal into terciles (the panel is too thin for quintiles), long the top / short
  the bottom equal-weight, earn the **next** month's return (one execution lag). The decisive
  statistic is the **Newey-West (HAC, Bartlett) t** of the monthly long-short series — the
  autocorrelation-robust bar `REAL` is written against (METHODOLOGY → *The inference bar*). A
  one-sample t and a monthly hit-rate accompany it.
- **Cross-check — pooled event drift.** The sibling-534 machinery: bucket all (ticker, filing)
  events by the signal, measure top-minus-bottom forward drift over ≈1m/1q/2q horizons, with a
  one-sample t and a **label-shuffle placebo** (permute signals, re-form random terciles).
- **Third axis — does it actually lead sales?** A pooled OLS of **next quarter's** YoY revenue
  growth on the current deferred-revenue growth (slope, t, R², correlation) plus the
  future-sales-growth spread between the top and bottom deferred-growth terciles. This is the
  *mechanism* check: the accounting lead can exist even if the stock does not move (and vice
  versa). The pooled t is read as suggestive — filings cluster by quarter, so it is not a
  calendar-robust HAC statistic.
- **Costs & borrow.** The tradability timer charges one-way cost × NAV × monthly turnover on
  **both** legs and makes the short leg pay an annualised borrow — the standard desk long-short
  friction model.
- **Coverage is a first-class caveat, not a footnote.** The deep-history names (VRSN, ADBE,
  ORCL, INTU, FICO…) begin ~2009; the pure-play SaaS names (SNOW, DDOG, CRWD, NET, ZM…) only
  IPO'd 2018-2020, so the monthly cross-section is small (single digits) before ~2012 and only
  becomes reasonably wide after ~2019. Terciles on a thin cross-section are noisy by
  construction; every number here should be read in that light.

## Survivorship — named on the Signal axis

The basket is **current survivors** (all still listed): a fixed roster of ~40 subscription-type
names that report the concept today. It cannot include subscription firms that were acquired or
failed. For a long-*top*/short-*bottom* deferred-*growth* signal both legs are drawn from the
same survivor pool, so the first-order equity-survivorship tilt partly cancels; the residual
risk is that *growth-signal informativeness* itself is survivor-conditioned (firms whose
deferred-revenue growth "worked out" are the ones still here). We therefore reason about the
bias direction explicitly rather than claiming it away, and never cite the survivor panel to
certify magnitude.

## Data sources

- **Deferred revenue / contract liabilities, revenue, total assets** — SEC EDGAR XBRL
  `companyconcept` API (`data.sec.gov`), 10-Q/10-K instant/duration facts, de-duplicated on
  period end (earliest filing wins), keeping the filing date so the signal is strictly
  point-in-time. Cached under `_cache/dr_events.csv`.
- **Daily adjusted closes** — yfinance (no key), cached under `_cache/dr_prices.csv`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [199-sales-growth](../199-sales-growth/) — ranks names on **recognised** annual **sales
  growth** (the income-statement number) and tests the LSV "high growth → low future return"
  premium. This study ranks on the **balance-sheet** deferred-revenue balance that *precedes*
  the sales print — a leading, off-income-statement signal, not the realised top line.
- [534-revenue-surprise-drift](../534-revenue-surprise-drift/) — the **post-revenue-surprise
  drift**: a *seasonal-random-walk surprise* in the **recognised revenue** figure and the drift
  after the print. That is an income-statement *surprise* event study; this is a
  *balance-sheet-level* characteristic sort on deferred revenue, which leads the recognised
  number by quarters.
- [799-order-backlog-drift](../799-order-backlog-drift/) — the sibling built alongside this one:
  **order backlog / remaining performance obligations** (signed but wholly-unbilled demand).
  Backlog sits one step *earlier* in the cash cycle than deferred revenue (backlog → billing →
  deferred revenue → recognised revenue). Same "intake leads the print" family, a different
  balance-sheet line item and a different disclosure (RPO vs contract liabilities).

None of the siblings rank on the **deferred-revenue / contract-liability balance growth** —
this study's own axis.
