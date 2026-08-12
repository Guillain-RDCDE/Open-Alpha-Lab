# References & literature map — Study 862 (Real Earnings Management)

## The claim under test

- **The thesis (Roychowdhury 2006).** When a firm is at risk of *missing* a benchmark — the prior
  year, zero, or the analyst consensus — it can massage the reported number not only through
  accruals (booking choices) but through **real operating decisions** that have genuine cash-flow
  consequences: (i) **overproduction** — build more units than demand warrants so fixed
  manufacturing overhead is spread across more units, lowering reported cost of goods sold per
  unit and inflating gross margin; and (ii) **cutting discretionary expenditure** — slashing R&D,
  SG&A and advertising to lift current operating income. The observable fingerprints are an
  **abnormally HIGH production cost** (`PROD = COGS + ΔInventory`) and an **abnormally LOW
  discretionary expense** (`DISX = R&D + SG&A`), each measured against a *normal-operations*
  benchmark. Roychowdhury (2006, *Journal of Accounting and Economics* 42) estimates the normal
  level as a fitted cross-sectional (industry-year) regression and calls the residual the
  *abnormal* (real-management) component.
- **The models.** Normal discretionary expense:
  `DISX_t/A_{t-1} = a0 + a1·(1/A_{t-1}) + a2·(Sales_{t-1}/A_{t-1}) + e` — regressed on **lagged**
  sales precisely because a firm that manages earnings *by boosting current sales* would otherwise
  look like it cut expenses. Normal production cost:
  `PROD_t/A_{t-1} = a0 + a1·(1/A_{t-1}) + a2·(Sales_t/A_{t-1}) + a3·(ΔSales_t/A_{t-1}) +
  a4·(ΔSales_{t-1}/A_{t-1}) + e`. The abnormal pieces are the residuals; the aggregate real-
  earnings-management proxy is `REM = ab_PROD − ab_DISX` (higher ⇒ more real management).
- **The subsequent-performance strand.** The reason a desk cares: real earnings management is
  *value-destroying* (you overproduce inventory you must later discount; you starve R&D that
  drives future growth), so firms that lean on it should show **lower subsequent operating
  performance and lower future returns**. This is documented by **Cohen, Dey & Lys (2008, *TAR*)**
  (REM rose after Sarbanes-Oxley as accrual management got riskier), **Cohen & Zarowin (2010,
  *JAE*)** (post-SEO underperformance is worse for real than accrual managers), and **Gunny (2010,
  *CAR*)** (real management around benchmarks and future ROA). **Li (2010)** and later work test
  whether the market *prices* REM in a timely way — the open question we take to the tape.
- **The open question we test.** On a small, honestly-thin panel of large US manufacturers that
  disclose the needed line items on EDGAR, does the REM proxy — ranked strictly on point-in-time
  filed values, held one execution lag, and charged realistic long-short costs plus borrow —
  earn a forward-return spread (the mispricing claim), and does it foreshadow a **gross-margin
  reversal** (the operating-mechanism claim)?

## What we measure, and the honesty rails

- **Signal, point-in-time.** `rem = ab_prod − ab_disx`, plus the two components `ab_prod` and
  `ab_disx` carried separately, computed from a single 10-Q/10-K and stamped with the **filing
  date** (`filed`), never the period end. Overproduction ⇒ high `ab_prod`; expense-cutting ⇒
  low `ab_disx` ⇒ high `−ab_disx`; both push REM up.
- **Primary test — calendar-time long-short (Newey-West).** Each month-end, rank the names
  carrying a fresh REM signal into terciles (the panel is too thin for quintiles), long the top /
  short the bottom equal-weight, earn the **next** month's return (one execution lag). The
  decisive statistic is the **Newey-West (HAC, Bartlett) t** of the monthly long-short series —
  the autocorrelation-robust bar `REAL` is written against. Because the literature predicts
  high-REM firms *under*-perform, a **negative** significant long-short would *support* the claim
  (a "buy the clean firms, short the manipulators" trade); a **positive** significant one would be
  a wrong-sign refutation.
- **Cross-check — pooled event drift.** Bucket all (ticker, filing) events by the signal, measure
  top-minus-bottom forward drift over ≈1m/1q/2q horizons, with a one-sample t and a **two-sided
  label-shuffle placebo** (permute signals, re-form random terciles). Two-sided because the sign
  is a hypothesis, not an assumption.
- **Third axis — the operating reversal.** A pooled OLS of the **forward change in gross margin**
  (`next_gm − gm`) on REM, plus the forward-Δgm spread between the top and bottom REM terciles.
  Overproduction inflates current gross margin and must reverse; expense-cutting borrows from the
  future. This is the *mechanism* check — the operating consequence can exist even if the stock
  does not move. The pooled t is read as suggestive: filings cluster by quarter, so it is not a
  calendar-robust HAC statistic.
- **Costs & borrow.** The tradability timer charges one-way cost × NAV × monthly turnover on
  **both** legs and makes the short leg pay an annualised borrow — the standard desk long-short
  friction model; the friction always shrinks the magnitude of the (possibly negative) edge.
- **Two modelling caveats stated up front.** (1) The normal-expense **benchmark coefficients are
  fit on the full pooled panel** — a mild in-sample look-ahead in the *benchmark*, not the signal
  value, and standard in a literature that fits contemporaneous industry-year cross-sections; the
  cross-sectional *rank* is what the sort trades. (2) With ~44 names we **pool across industries**
  rather than run Roychowdhury's per-industry-year regressions, so `ab_prod`/`ab_disx` carry
  industry-composition noise. Both bias toward *finding nothing cleaner*, not toward a false
  positive.
- **Coverage is a first-class caveat, not a footnote.** XBRL quarterly *flow* tags are sparse for
  the **fiscal fourth quarter** (10-Ks disclose the full year, not Q4), R&D is missing for some
  consumer names, and the pre-2009 XBRL era is thin — so the usable cross-section is smaller than
  the roster and gappy. Terciles on a thin cross-section are noisy by construction; every number
  here should be read in that light.

## Survivorship — named on the Signal axis

The basket is **current survivors** (all still listed): a fixed roster of ~44 large manufacturers
/ hardware / pharma / industrials that report the concepts today. It cannot include firms that
were acquired or failed. For a long-*top*/short-*bottom* REM sort both legs are drawn from the
same survivor pool, so the first-order equity-survivorship tilt partly cancels; the residual risk
is that *REM's informativeness* is itself survivor-conditioned (firms whose real management "did
not sink them" are the ones still here). We reason about the bias direction explicitly rather than
claiming it away, and never cite the survivor panel to certify magnitude.

## Data sources

- **Revenues, CostOfRevenue / CostOfGoodsAndServicesSold, SG&A, ResearchAndDevelopmentExpense**
  (flows) and **InventoryNet, Assets** (instants) — SEC EDGAR XBRL `companyconcept` API
  (`data.sec.gov`), 10-Q/10-K facts, de-duplicated on period end (earliest filing wins), keeping
  the filing date so the signal is strictly point-in-time. Cached under `_cache/rem_events.csv`.
- **Daily adjusted closes** — yfinance (no key), cached under `_cache/rem_prices.csv`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [574-penny-beat](../574-penny-beat/) — the tendency of firms to **just beat** EPS by a penny (a
  distribution-of-surprises discontinuity). That documents the *outcome* (a suspicious kink at
  zero surprise); this study builds the **real-activity mechanism** (production + discretionary
  expense residuals) that is *one way* firms achieve such beats, and tests its forward return.
- [229-beneish-m-score](../229-beneish-m-score/) — the Beneish **M-score**, an
  eight-variable **accrual / manipulation-likelihood** classifier built from balance-sheet and
  income-statement ratios (DSRI, GMI, AQI, …). That detects *accounting* (accrual) manipulation;
  REM is the deliberately-*non*-accrual sibling — real operating choices that leave clean books.
- [855-accrual-quality](../855-accrual-quality/) — the Dechow-Dichev **accrual-quality** /
  earnings-persistence signal, i.e. how well accruals map to cash flows. Again an *accrual*
  construct; REM is precisely the manipulation channel that accrual-quality metrics **miss** by
  design.
- [525-r-and-d-intensity](../525-r-and-d-intensity/) — ranks firms on the **level** of R&D/sales
  (the R&D-anomaly / intangible-intensity premium). This study uses R&D only as one ingredient of
  the *abnormal discretionary-expense residual* (a within-firm cut relative to its normal level),
  not as a cross-firm intensity characteristic.

None of the siblings rank on the **Roychowdhury abnormal-production-plus-abnormal-discretionary-
expense** proxy — this study's own axis.
