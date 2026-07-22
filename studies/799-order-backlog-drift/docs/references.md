# References & literature map — Study 799 (Order-Backlog Drift, RPO)

## The claim under test

- **The folklore / thesis.** *"Order backlog is a leading indicator the market is slow to
  price."* A firm's backlog — contracted work signed but not yet delivered — mechanically
  front-runs the income statement: sign a multi-year enterprise contract today, and it
  bleeds into reported revenue over the coming quarters. If backlog is compounding faster
  than sales, the argument goes, future revenue (and the stock) should follow, and a market
  that fixates on the trailing income statement *underreacts* to the balance-sheet signal.
- **The modern, machine-readable proxy.** Since **ASC 606** (FASB ASU 2014-09, effective for
  fiscal years beginning after 2017-12-15), US filers must disclose their **Remaining
  Performance Obligations (RPO)** — the aggregate transaction price allocated to
  unsatisfied (or partially unsatisfied) performance obligations. RPO is the clean, XBRL-
  tagged successor to the old free-text "backlog" paragraph, tagged
  `us-gaap:RevenueRemainingPerformanceObligation`. It is now a headline KPI for enterprise-
  software investors (Snowflake, CrowdStrike, ServiceNow, Palantir all guide on it).
- **The academic anchor.** The idea that a *known* fundamental predicts returns because the
  market underreacts is the post-earnings-announcement-drift (PEAD) tradition — Ball &
  Brown (1968); Bernard & Thomas (1989, 1990, *Evidence that stock prices do not fully
  reflect the implications of current earnings for future earnings*). On order backlog
  specifically: **Rajgopal, Shevlin & Venkatachalam (2003, "Does the stock market fully
  appreciate the implications of leading indicators for future earnings? Evidence from
  order backlog," *Review of Accounting Studies*)** document a backlog-based return
  predictability consistent with underreaction. **Lev & Thiagarajan (1993)** include
  order-backlog growth among their twelve "fundamental signals." RPO is the post-2018,
  standardised, cross-sector version of that same backlog variable.
- **The open question we test.** Does ranking a fixed basket of RPO-disclosing enterprise-
  software names on **YoY RPO growth** produce a forward-return long-short that clears the
  desk's autocorrelation-robust bar — on the short, one-regime, post-2018 sample that is
  *all the RPO history that exists*?

## What we measure, and the honesty rails

- **Primary — calendar-time tercile long-short.** Each month-end, rank the names carrying a
  fresh RPO-YoY-growth figure into terciles; long the top, short the bottom, equal-weight;
  earn the next month's return. **One documented execution lag** (signal known at month *t*'s
  close, return of *t+1*). The decisive statistic is the **Newey-West (1987)** HAC *t* of
  the monthly long-short series — the serial-correlation-robust number the `REAL` bar is
  written against.
- **Cross-check — pooled event drift (534-style).** Bucket every (ticker, filing-date) event
  by the signal; measure the top-minus-bottom forward drift over 1m / 1q / 2q horizons,
  entered **one session after the filing date** (the day the RPO number became public);
  one-sample *t* + a 10,000-draw label-shuffle placebo.
- **Mechanism (3rd axis) — leads sales?** A pooled OLS of next-quarter revenue YoY growth on
  this-quarter RPO YoY growth (slope, *t*, R², tercile spread). Even if the stock doesn't
  move, does the accounting lead actually exist? (The *t* is iid-pooled — events cluster by
  calendar quarter — so it is read as suggestive, not a calendar-robust HAC statistic.)
- **Point-in-time, no look-ahead.** Every signal is dated to the **filing date** of the
  10-Q/10-K that disclosed the RPO, and every forward return / sales figure is measured
  strictly after it. Same-period-end figures reported in multiple filings keep the
  earliest disclosure (no restatement look-ahead).
- **Costs.** One-way × NAV × turnover on **both** legs, plus **borrow on the short leg**
  (100 bps annualised); gross and net are labelled separately everywhere.
- **Survivorship named on the Signal axis.** The basket is current survivors (all still
  trading), so acquired/failed software names are absent. For a long-top/short-bottom
  *growth* signal the direction is ambiguous (both legs are survivors), but it can only be
  argued away, not ignored — and on this thin panel it is a live caveat, not a footnote.

## Why coverage is thin (stated as a decision, not hidden)

RPO **did not exist before ASC 606**. Essentially no name in the panel reports it before
2018, and the cross-section is only wide enough to sort into terciles from ~2019. So the
whole study lives inside a **single ~7-year macro regime** (the 2020 crash, the 2021 SaaS
melt-up, the 2022 rate-shock de-rating, and the 2023-25 recovery), on a slow quarterly
signal that refreshes ~4×/year per name. That is a small effective sample for a HAC *t* by
construction — the honest prior is that this tape *cannot certify* a subtle underreaction
effect even if one exists, which is exactly why the synthetic positive control is carried:
to prove the machinery can bank a planted edge when there is one.

## Data sources

- **RPO** — SEC EDGAR XBRL `companyconcept` API (`data.sec.gov`), us-gaap concept
  `RevenueRemainingPerformanceObligation` (an instant balance), with period-end and filing
  date. Revenue (`RevenueFromContractWithCustomerExcludingAssessedTax` / `Revenues`) and
  `Assets` for the mechanism and the scaled variant. Cached under `_cache/rpo_events.csv`.
- **Prices** — yfinance daily adjusted close for the basket, cached under
  `_cache/rpo_prices.csv`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [798-deferred-revenue-signal](../../798-deferred-revenue-signal/) — the **sibling**, and
  the closest one. It ranks on YoY growth in **deferred revenue / contract liabilities**
  (`DeferredRevenueCurrent` / `ContractWithCustomerLiabilityCurrent`) — the *billed*-but-
  unrecognised balance. RPO is a **different, broader line item**: it is *contracted* but
  not-yet-billed *and* not-yet-recognised — the full signed backlog, including future
  invoicing a deferred-revenue balance has not captured yet. Same underreaction family, a
  strictly larger and more forward-looking accounting quantity.
- [199-sales-growth](../../199-sales-growth/) — ranks on **realised** trailing sales growth,
  the *income-statement* number. This study's whole premise is that RPO **leads** that
  realised number, so ranking on backlog growth is meant to be earlier information than
  ranking on sales growth.
- [534-revenue-surprise-drift](../../534-revenue-surprise-drift/) — a **revenue-surprise**
  PEAD study (reported revenue vs a seasonal-random-walk expectation, then drift). It is a
  *surprise* around the income statement's top line; this study is a *level-growth* signal
  on the backlog that precedes the top line, sorted in calendar time rather than around the
  announcement.

None of the siblings rank on the **RPO / signed-backlog** growth itself — that is this
study's own axis.

## Shared method citations

- Newey, W. & West, K. (1987). *A simple, positive semi-definite, heteroskedasticity and
  autocorrelation consistent covariance matrix.* Econometrica.
- Wilson, E. B. (1927). *Probable inference, the law of succession, and statistical
  inference.* JASA (the score interval on the hit rate).
- Bernard, V. & Thomas, J. (1989/1990). PEAD / underreaction — the family this belongs to.
