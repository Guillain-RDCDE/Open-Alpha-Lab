# References & literature map — Study 853 (Days-Sales-Outstanding Buildup)

## The claim under test

- **The thesis.** **Days Sales Outstanding (DSO)** measures how many days of sales are tied up in
  unpaid customer invoices: `DSO = AccountsReceivableNetCurrent / (Revenues / 365)`. When a firm's
  **receivables grow faster than its sales**, DSO rises — and the fundamental-analysis literature
  reads a rising DSO as a **red flag**: it can mean *channel-stuffing* (pushing product into the
  distribution channel on generous credit terms to make the quarter, booked as revenue with a
  matching receivable), *aggressive revenue recognition*, or simply *deteriorating collections*.
  The prediction is a **negative** one — high DSO *buildup* precedes *weaker* future earnings and
  returns — so the tradeable expression is **long the low-DSO-change names, short the
  high-DSO-change names**.
- **The academic anchor.** The receivables signal is one of the nine fundamental signals in
  **Abarbanell & Bushee (1997, "Fundamental Analysis, Future Earnings, and Stock Prices",
  *Journal of Accounting Research*; 1998, *The Accounting Review*)**, whose "accounts receivable"
  component is exactly `%ΔReceivables − %ΔSales` — the sign of a rising DSO. It is a cousin of
  **Sloan (1996, *The Accounting Review*)** on the mispricing of the accrual vs cash components of
  earnings (a receivable *is* an accrual), of **Lev & Thiagarajan (1993, *JAR*)** on
  fundamental-signal value-relevance, and it overlaps the receivables term inside the
  **Piotroski (2000) F-score** and **Beneish (1999) M-score** (the DSRI ratio). The channel-
  stuffing failure mode is the textbook forensic-accounting warning.
- **The open question we test.** On a small, honestly-thin panel of large US filers that report
  both a current-receivables balance and quarterly revenue on EDGAR, does the **year-over-year
  change in DSO** (a) **warn on future sales** (the channel-stuffing mechanism) and (b) **earn a
  forward return spread** in the claimed direction (low-buildup minus high-buildup), once you rank
  strictly on point-in-time filed values, hold with one execution lag, and charge realistic
  long-short costs plus borrow?

## What we measure, and the honesty rails

- **Signal, point-in-time.** `dso_yoy_chg` = current-quarter DSO minus the same quarter a year
  earlier, in **days**, known only at the **10-Q/10-K filing date** (`filed`), never the period
  end. A unit-free percentage-change variant (`dso_yoy_pct`) is carried as a robustness cut. The
  ranking column `dso_score = −dso_yoy_chg` is signed so that HIGH score = LOW buildup = the
  attractive (long) side; a *wrong-sign* result surfaces as a *negative* long-short. Concept:
  `AccountsReceivableNetCurrent` with a fallback to `ReceivablesNetCurrent`; revenue is `Revenues`
  falling back to `RevenueFromContractWithCustomerExcludingAssessedTax` (longest per-name history).
  One quarter of revenue is annualised to a daily run-rate; the annualisation factor and
  seasonality cancel in the same-fiscal-quarter YoY change.
- **Primary test — calendar-time long-short (Newey-West).** Each month-end, rank the names that
  carry a fresh signal into terciles (the panel is too thin for quintiles), long the low-buildup
  third / short the high-buildup third equal-weight, earn the **next** month's return (one
  execution lag). The decisive statistic is the **Newey-West (HAC, Bartlett) t** of the monthly
  long-short series — the autocorrelation-robust bar `REAL` is written against (METHODOLOGY → *The
  inference bar*). A one-sample t and a Wilson-interval monthly hit-rate accompany it.
- **Cross-check — pooled event drift.** Bucket all (ticker, filing) events by the signed score,
  measure top-minus-bottom forward drift over ≈1m/1q/2q horizons, with a one-sample t and a
  **label-shuffle placebo** (permute signals, re-form random terciles).
- **Third axis — does the buildup warn on sales?** A pooled OLS of **next quarter's** YoY revenue
  growth on this quarter's DSO change (slope, t, R², correlation) plus the future-sales-growth
  spread between the low- and high-buildup terciles. The channel-stuffing story predicts a
  **negative** slope. This is the *mechanism* check; the pooled t is read as suggestive — filings
  cluster by quarter, so it is not a calendar-robust HAC statistic.
- **Costs & borrow.** The tradability timer charges one-way cost × NAV × monthly turnover on
  **both** legs and makes the short leg pay an annualised borrow — the standard desk long-short
  friction model.
- **Coverage is a first-class caveat, not a footnote.** Only a subset of the basket exposes a long,
  cleanly-matched (receivables, revenue) history on EDGAR XBRL; per-name revenue tagging changed
  with ASC 606 (2018) and some names switch concepts mid-history, so the matched cross-section is
  smaller than the raw name count and thin early in the sample. Terciles on a thin cross-section
  are noisy by construction; every number here should be read in that light.

## Survivorship — named on the Signal axis

The basket is **current survivors** (all still listed): a fixed roster of ~48 large US filers with
genuine trade receivables that report the concept today. It cannot include firms that were
acquired or delisted. For a long-*low-change*/short-*high-change* signal both legs are drawn from
the same survivor pool, so the first-order equity-survivorship tilt partly cancels; the residual
risk is that *signal informativeness* itself is survivor-conditioned. We therefore reason about the
bias direction explicitly rather than claiming it away, and never cite the survivor panel to
certify magnitude.

## Data sources

- **Accounts receivable (current) and quarterly revenue** — SEC EDGAR XBRL `companyconcept` API
  (`data.sec.gov`), 10-Q/10-K instant/duration facts, de-duplicated on period end (earliest filing
  wins), keeping the filing date so the signal is strictly point-in-time. Cached under
  `_cache/dso_events.csv`. CIKs are resolved from the SEC company-ticker map.
- **Daily adjusted closes** — yfinance (no key), cached under `_cache/dso_prices.csv`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [231-sloan-accruals](../231-sloan-accruals/) — the **total-accruals** anomaly (Sloan 1996): the
  balance-sheet change in net working capital less depreciation, scaled by assets. Receivables are
  *one* accrual line; this study isolates the **receivables-vs-sales** ratio (DSO) specifically,
  not the aggregate accrual.
- [522-percent-operating-accruals](../522-percent-operating-accruals/) — **percent operating
  accruals** (accruals scaled by earnings, the Hafzalla-Lundholm-Van Winkle cut). Again an
  aggregate-accrual construction; DSO is a single, interpretable *days* ratio with a specific
  channel-stuffing narrative.
- [529-inventory-growth](../529-inventory-growth/) — **inventory growth**, the *other* working-
  capital red flag (goods piling up unsold). DSO is the **receivables** side of the same
  working-capital story; the two are complementary forensic signals, not the same number.
- [855-accrual-quality](../855-accrual-quality/) — **accrual quality** (Dechow-Dichev: how well
  accruals map into realised cash flows). That grades the *reliability* of accruals in aggregate;
  this study ranks on the *level of buildup* in one specific accrual (receivables) via DSO.

None of the siblings rank on the **year-over-year change in Days Sales Outstanding** itself — this
study's own axis.
