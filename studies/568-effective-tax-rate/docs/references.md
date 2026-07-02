# References & literature map — Study 568 (Effective-Tax-Rate)

## The claim, at full strength (and why the sign is contested)

- **Thomas & Zhang (2011)**, *"Tax Expense Momentum."* *Journal of Accounting Research* 49(3).
  Seasonal changes in tax expense predict future returns and earnings surprises — the market
  under-reacts to the information in the *tax* line. The empirical seed that ETR-based signals
  carry return-relevant information.
- **Weber (2009)**, *"Do Analysts and Investors Fully Appreciate the Implications of Book-Tax
  Differences for Future Earnings?"* *Contemporary Accounting Research* 26(4). Large book-tax
  differences (a cousin of a low ETR) predict lower future earnings and returns — the **red-flag /
  risk** reading of aggressive tax positions.
- **Hanlon (2005)**, *"The Persistence and Pricing of Earnings, Accruals, and Cash Flows When Firms
  Have Large Book-Tax Differences."* *The Accounting Review* 80(1). Firms with large book-tax
  differences have *less persistent* earnings — the mechanism behind the fragile-loophole story.
- **Katz, Khan & Schmidt (2013)** / **Desai & Dharmapala (2009)**, on tax avoidance and firm value:
  the **quality / tax-avoidance premium** reading — efficient, well-governed avoiders can be
  *rewarded* — the opposite-signed hypothesis this study pits against the red-flag story.
- **Dyreng, Hanlon & Maydew (2008)**, *"Long-Run Corporate Tax Avoidance."* *The Accounting
  Review* 83(1). Long-run (multi-year) cash ETRs vary enormously across firms and are persistent —
  motivating the *level* as a firm characteristic, and the caution that a one-year ETR is noisy.

The literature does **not** agree on the sign — which is exactly why this study reports the
low-minus-high-ETR hedge, its placebo null, its IC and its window-stability, and lets the tape
adjudicate rather than assuming a direction.

## The signal we build

- **Effective tax rate**: `ETR = income_tax_expense / pretax_income`, computed only where pretax
  income is comfortably positive (a loss-making firm has no economic tax *rate*), winsorised to a
  sane band. Built from EDGAR annual 10-K FY facts (`IncomeTaxExpenseBenefit` over the
  `IncomeLossFromContinuingOperationsBeforeIncomeTaxes…` concept family). We also compute the
  **change in ETR** (Δ ETR) as a second sort key, per the tax-expense-momentum reading. The full
  academic version uses a multi-year *cash* ETR and industry adjustment; we name those
  simplifications on the SIGNAL axis.

## Neighbours on this bench (the dedup map)

- **[Study 192 — Tax-Day](../../192-tax-day/)** — a *calendar* seasonal around the April tax
  deadline (retirement-account flows). Study 568 is the firm-level **ETR return anomaly**, nothing
  to do with the calendar.
- **[Study 122 — Gross-Profitability](../../122-gross-profitability/)** /
  **[Study 200 — ROE-Quality](../../200-roe-quality/)** — profitability/quality sorts scaling
  earnings by assets or equity. Study 568 sorts on the **tax line** specifically (level and change),
  a different characteristic that overlaps quality only loosely.
- **[Study 231 — Sloan-Accruals](../../231-sloan-accruals/)** /
  **[Study 521/522 — cash-profitability & percent-accruals](../../522-percent-operating-accruals/)** —
  accrual/cash-earnings anomalies. Shared *machinery* (annual quintile sort, placebo null, EDGAR +
  yfinance panel) but a different signal; Study 568 reuses that method template.

## Shared method

- **HAC (Newey-West 1987)** standard errors — the *t*-stat on the annual hedge and IC series.
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null: permute the
  ETR labels within each year and read the hedge-*t*'s tail probability.
- **Information coefficient** (Grinold & Kahn) — the year-by-year cross-sectional rank correlation
  between the signal and the next-year return.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a robust
  *t* ≥ 2 on the real tape, a placebo null, seed-robustness), the explicit survivorship caveat, one
  execution lag (fiscal year y → calendar year y+1), and costs one-way × NAV with shorts paying
  borrow.
