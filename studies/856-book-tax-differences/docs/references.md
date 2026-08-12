# References & literature map — Study 856 (Book-Tax Differences)

## The claim under test

- **The thesis.** A firm's **book income** (pretax accounting income on the income statement) and
  its **taxable income** (what it actually owes tax on) are computed under two different rule
  books, so they diverge. When book income sits *far above* the taxable income *implied* by the
  reported tax expense — a **large positive book-tax difference (BTD)** — it is argued to be a red
  flag: the accounting earnings are being propped up by items that do not (yet) show up in taxable
  income, so those earnings are **less likely to persist**, and a market slow to see it should
  earn **lower future returns** on the high-BTD names. The tradeable read-through: **long low-BTD
  ("clean") firms, short high-BTD ("aggressive") firms**.
- **The academic anchor.** The canonical result is **Michelle Hanlon (2005), "The Persistence and
  Pricing of Earnings, Accruals, and Cash Flows When Firms Have Large Book-Tax Differences", *The
  Accounting Review* 80(1)**. Hanlon shows firm-years with large (positive *or* negative) book-tax
  differences have **less persistent earnings** and that the market appears not to fully price
  this — a mispricing in the Sloan (1996, *The Accounting Review*) accruals-anomaly family. The
  book-tax gap is closely related to the **deferred-tax expense** signal of **Lev & Nissim (2004,
  *The Accounting Review*)** and to tax-avoidance measures surveyed in **Hanlon & Heitzman (2010,
  *Journal of Accounting and Economics*)**. We test the **strong, tradeable form**: rank names on
  the scaled book-tax gap and measure the forward long-short return spread, plus the persistence
  mechanism directly.
- **How we compute it.** We do not observe taxable income (it is confidential), so — as in the
  literature — we **impute** it by grossing up the current income-tax expense through the
  **statutory** rate: implied taxable income = `IncomeTaxExpenseBenefit / statutory_rate`, and
  `BTD = PretaxIncome − implied_taxable_income`, scaled by total assets. Grossing up by the
  *statutory* rate (not the firm's *effective* rate) is deliberate: dividing by the effective rate
  would define the gap away (it would just recover pretax income). The statutory rate is
  time-varying: **35 %** through fiscal-2017 and **21 %** from fiscal-2018 (the Tax Cuts and Jobs
  Act, effective for tax years beginning after 2017-12-31). We ignore the 2018 **blended-rate**
  subtlety for off-calendar fiscal years — a documented simplification that affects a single
  transition year.
- **The open question we test.** On an honestly-thin panel of large US filers, does the book-tax
  gap (a) **predict a forward return spread** (the mispricing claim) and (b) **mark less-persistent
  earnings** (the mechanism), once you rank strictly on point-in-time filed values, hold with one
  execution lag, and charge realistic long-short costs plus borrow?

## What we measure, and the honesty rails

- **Signal, point-in-time.** `btd_assets` = (Pretax − Tax / statutory-rate) / Assets, from the
  annual 10-K, known only at the **filing date** (`filed`), never the fiscal-year end. The ranking
  signal is `btd_neg = −btd_assets` so the engine's "top tercile, long" is the *lowest* gap. A
  year-on-year **change** variant `−ΔBTD/Assets` is carried as a robustness cut.
- **Primary test — calendar-time long-short (Newey-West).** Each month-end, rank the names carrying
  a fresh (≤ ~14-month-old) signal into terciles (the panel is too thin for quintiles), long the
  low-BTD tercile / short the high-BTD tercile equal-weight, earn the **next** month's return (one
  execution lag). The decisive statistic is the **Newey-West (HAC, Bartlett) t** of the monthly
  long-short series — the autocorrelation-robust bar `REAL` is written against (METHODOLOGY → *The
  inference bar*).
- **Cross-check — pooled event drift.** Bucket all (ticker, filing) events by the signal, measure
  top-minus-bottom forward drift over ≈1q/2q/1y horizons, with a one-sample t and a **label-shuffle
  placebo** (permute signals, re-form random terciles), plus the tercile monotonicity picture.
- **Third axis — does it mark less-persistent earnings?** The Hanlon *mechanism*: an interaction
  OLS of next-year pretax ROA on this-year ROA, letting the persistence slope differ between the
  highest- and lowest-BTD terciles. A **negative** interaction (high-BTD earnings persist less) is
  the effect Hanlon documents. The pooled t is read as suggestive — firm-years cluster — so we cite
  the magnitude of the persistence gap, not the literal t.
- **Costs & borrow.** The tradability timer charges one-way cost × NAV × monthly turnover on
  **both** legs and makes the short leg pay an annualised borrow — the standard desk long-short
  friction model.
- **Coverage is a first-class caveat, not a footnote.** These are large-cap survivors — precisely
  the firms where book-tax differences are smallest and most-scrutinised, and where Hanlon's effect
  is *weakest*. The monthly cross-section runs from 6 names (2010) to 37, averaging ≈30. Any effect
  here is a conservative reading; the absence of one does not refute the broad-cross-section result.

## Survivorship — named on the Signal axis

The basket is **current survivors** (all still listed): a fixed roster of ~42 large-cap names with
deep EDGAR + yfinance histories. It cannot include firms whose aggressive tax positions blew up and
were delisted — exactly the tail where a red-flag effect should be strongest. So every real-tape
number here is an **upper bound on the noise, a lower bound on any true signal**, and is named on
the Signal axis. A survivor-only panel can never earn `REAL` — that needs a robust `t ≥ 2` on a
survivorship-free tape.

## Data sources

- **Pretax income, income-tax expense, total assets** — SEC EDGAR XBRL `companyconcept` API
  (`data.sec.gov`), annual 10-K duration/instant facts, de-duplicated on period end (earliest
  filing wins), keeping the filing date so the signal is strictly point-in-time. Cached under
  `_cache/btd_events.csv`.
- **Daily adjusted closes** — yfinance (no key), cached under `_cache/btd_prices.csv`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [568-effective-tax-rate](../../568-effective-tax-rate/) — ranks names on the **level** of the
  *effective* tax rate (`tax / pretax`) and its change. That is the tax rate itself; **this** study
  is the **book-tax GAP** — the dollar difference between book income and the income the tax bill
  implies, scaled by assets. A low ETR and a large BTD are related but distinct (a firm can pay a
  low rate with no book-tax gap, or a statutory rate with a large gap from timing differences).
- [231-sloan-accruals](../../231-sloan-accruals/) — the **total-accruals** anomaly (Sloan 1996):
  the accrual vs cash component of earnings. The book-tax difference is a *specific, tax-flavoured*
  slice of the same accruals family (Hanlon situates her result inside Sloan's), but it is
  constructed from the tax footnote, not from working-capital accruals.
- [229-beneish-m-score](../../229-beneish-m-score/) — a multi-ratio **earnings-manipulation**
  detector. It blends eight accounting signals into a manipulation score; the book-tax difference
  is one *single, tax-specific* red flag, tested here in isolation on its own return and
  persistence claims.

None of the siblings rank on the **statutory-grossed-up book-minus-tax income difference** — this
study's own axis.
