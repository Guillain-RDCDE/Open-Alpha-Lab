# References — Study 631 (Restatement Shock, Item 4.02)

## The claim's source

- **SEC Release No. 33-8400 (2004)** — *Additional Form 8-K Disclosure Requirements and
  Acceleration of Filing Date*. Created **Item 4.02**, "Non-Reliance on Previously Issued
  Financial Statements or a Related Audit Report" (effective 2004-08-23) — the "do not rely"
  confession this study events on. <https://www.sec.gov/rules/final/33-8400.htm>
- The folk claim under test — *"restatement announcements keep hurting for months; the market
  underreacts to accounting bombs"* — is the accounting-restatement cousin of post-event drift
  (PEAD-style underreaction), repeated across practitioner forensic-accounting writing.

## Key papers

- **Palmrose, Z-V., V. Richardson & S. Scholz (2004)** — *Determinants of market reactions to
  restatement announcements*, **Journal of Accounting and Economics** 37(1). The canonical
  announcement-effect estimate: ≈ **−9%** two-day abnormal return on average, worse for fraud
  and auditor-initiated restatements. <https://doi.org/10.1016/j.jacceco.2003.06.003>
- **Hribar, P. & N. Jenkins (2004)** — *The effect of accounting restatements on earnings
  expectations and estimated cost of capital*, **Review of Accounting Studies** 9. Restatements
  raise the implied cost of capital persistently — a months-long repricing, not a one-day hit.
  <https://doi.org/10.1023/B:RAST.0000028194.11371.42>
- **Files, R., E. Swanson & S. Tse (2009)** — *Stealth disclosure of accounting restatements*,
  **The Accounting Review** 84(5). Reaction (and subsequent drift) depends on disclosure
  prominence — low-prominence restatements get repriced late, direct underreaction evidence.
  <https://doi.org/10.2308/accr.2009.84.5.1495>
- **GAO (2002, 2006)** — *Financial Statement Restatements* reports (GAO-03-138, GAO-06-678):
  restatement frequency, and average market-cap losses far beyond the announcement window.
  <https://www.gao.gov/products/gao-06-678>
- **Audit Analytics** — annual *Financial Restatements* reviews: Item 4.02 ("reissuance")
  restatement counts peak mid-2000s (SOX 404 era) and decline steadily afterwards — the regime
  drift behind our per-era split. <https://www.auditanalytics.com/>
- **Bernard, V. & J. Thomas (1989)** — *Post-earnings-announcement drift: delayed price response
  or risk premium?*, **Journal of Accounting Research** 27. The template underreaction anomaly
  this claim borrows its logic from. <https://doi.org/10.2307/2491062>
- **Shleifer, A. & R. Vishny (1997)** — *The limits of arbitrage*, **Journal of Finance** 52(1).
  The third-axis mechanism: mispricing persists where arbitrage capital won't go (small,
  illiquid, hard-to-borrow names). <https://doi.org/10.1111/j.1540-6261.1997.tb03807.x>

## Siblings on this desk (dedup framing)

- [229-beneish-m-score](../../229-beneish-m-score/) — the Beneish M-Score **predicts**
  manipulation from reported accruals *before* anyone admits anything. This study is the other
  end of the pipeline: the **confession event** itself (the company files an 8-K saying the
  numbers were wrong) and what happens *after* the admission.
- [231-sloan-accruals](../../231-sloan-accruals/) and
  [522-percent-operating-accruals](../../522-percent-operating-accruals/) — accrual-quality
  cross-sections; no event, no confession.
- [369-earnings-revision-momentum](../../369-earnings-revision-momentum/) — underreaction to
  analyst news; same behavioral engine, different (non-forensic) trigger.

## Data sources

- **EDGAR full-text search** (`efts.sec.gov/LATEST/search-index?q="Item 4.02"&forms=8-K`) —
  the event sample: 8-K filings whose structured `items` field contains `4.02`, sampled per
  quarter 2004-Q4 → 2026-Q2. <https://efts.sec.gov/LATEST/search-index?q=%22Item+4.02%22&forms=8-K>
- **SEC `company_tickers.json`** — current CIK→ticker map (the *current* qualifier is the
  named deads-missing bias). <https://www.sec.gov/files/company_tickers.json>
- **yfinance** — daily adjusted (total-return) closes + volumes for matched tickers and SPY
  (the market benchmark). <https://github.com/ranaroussi/yfinance>

## Shared method citations

- Newey, W. & K. West (1987) — HAC standard errors (the clustered-by-month drift *t*).
- Welch, B. L. (1947) — unequal-variance two-sample *t* (the small-vs-large split).
- House rules: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar, one execution
  lag, costs × NAV with shorts paying borrow, survivorship named on the Signal axis.
