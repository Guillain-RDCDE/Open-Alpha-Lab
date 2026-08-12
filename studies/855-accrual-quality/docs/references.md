# References & literature map — Study 855 (Accrual Quality, Dechow-Dichev)

## The claim under test

- **The thesis.** Accruals exist to shift the timing of cash flows into reported earnings —
  a sale on credit is revenue today even though the cash lands next quarter. **Dechow & Dichev
  (2002)** argue that the *quality* of that timing job varies: **good** accruals map cleanly into
  the cash flows they anticipate, while **poor** accruals are dominated by estimation error and
  noise that later reverses. Their operational measure is the **standard deviation of the
  residual** from regressing working-capital accruals on **lagged, current and lead operating
  cash flow** — a large residual vol means accruals that do *not* track cash, i.e. low quality.
  The trading form (this study's strong claim): low-quality earnings are less persistent and,
  the argument goes, discounted, so a book **long high-quality (low residual vol) / short
  low-quality (high residual vol)** should earn a positive spread.
- **The academic anchor.** Dechow & Dichev (2002, *The Accounting Review*, "The Quality of
  Accruals and Earnings: The Role of Accrual Estimation Errors") define the residual-vol measure
  and show it correlates with earnings persistence. **Francis, LaFond, Olsson & Schipper (2005,
  *Journal of Accounting and Economics*, "The market pricing of accruals quality")** turn it into
  a cross-sectional **risk factor** — and, crucially, find that **poor** accrual quality carries
  a **higher** cost of capital / higher expected return (a *premium for bearing information
  risk*), not a discount. That is the direction tension we test head-on: the retail "buy quality"
  framing and the Francis "poor quality earns more" factor point *opposite* ways. It is a cousin
  of the broader **accrual anomaly** of **Sloan (1996)** — the market over-weights the accrual
  component of earnings — but it grades the *reliability* of accruals, not their *level* or sign.
- **The open question we test.** On an honestly-thin panel of deep-history US non-financial
  filers, does a point-in-time Dechow-Dichev residual-vol sort (a) **predict a positive
  high-minus-low-quality forward return** (the strong trading claim), and (b) **flag genuine
  earnings-persistence differences** (the DD validity claim), once we rank strictly on
  point-in-time filed values, hold with one execution lag, and charge realistic long-short costs
  plus borrow?

## What we measure, and the honesty rails

- **Signal, point-in-time.** `aq_vol` = the std of the OLS residual of
  `(NI−CFO)/avg-assets` on `[1, CFO_{t-1}, CFO_t, CFO_{t+1}]/avg-assets` over a rolling
  **12-quarter** window (min 8 usable quarters). We use the **cash-flow-statement total accrual**
  (`NI − CFO`) as the EDGAR-robust proxy for DD "current accruals"; a **balance-sheet
  working-capital** variant (residual vol of `(ΔReceivables + ΔInventory)/avg-assets`) is carried
  as a robustness cut. `quality = −aq_vol`, so a high-quality name has a *low* residual vol.
  The window at quarter *q*'s filing uses only quarters whose full (lag, current, lead) CFO triple
  is already public — the lead term never peeks past the filing date (no look-ahead).
- **Quarterly-flow reconstruction.** EDGAR/XBRL reports income and cash-flow facts **year-to-date**
  (3/6/9/12-month cumulative), not per quarter, so a naïve "90-day span" filter recovers barely
  one clean CFO point per firm-year. We reconstruct quarterly flows by **differencing the
  cumulative YTD chain** (facts sharing a fiscal-year start), keeping only ~one-quarter implied
  spans. This is the standard fix and is essential to getting a usable DD panel at all.
- **Primary test — calendar-time long-short (Newey-West).** Each month-end, rank the names that
  carry a fresh signal into terciles (the panel is too thin for quintiles), long the top
  (high quality) / short the bottom equal-weight, earn the **next** month's return (one execution
  lag). The decisive statistic is the **Newey-West (HAC, Bartlett) t** of the monthly long-short
  series — the autocorrelation-robust bar `REAL` is written against. A one-sample t and a monthly
  hit-rate accompany it.
- **Cross-check — pooled event drift.** Bucket all (ticker, filing) events by the signal, measure
  top-minus-bottom forward drift over ≈1m/1q/2q horizons, with a one-sample t and a **label-shuffle
  placebo** (permute signals, re-form random terciles), plus the tercile monotonicity picture.
- **Third axis — does it flag earnings quality?** Split into quality terciles and compare the
  **ROA persistence slope** (next-quarter ROA on current ROA) and the **earnings volatility** of
  the best- vs worst-quality tercile. This is the DD *validity* check — it can hold even if the
  stock does not move (and it does hold, decisively). The persistence t is read as suggestive —
  filings cluster by quarter, so it is not a calendar-robust HAC statistic.
- **Costs & borrow.** The tradability timer charges one-way cost × NAV × monthly turnover on
  **both** legs and makes the short leg pay an annualised borrow — the standard desk long-short
  friction model.
- **Coverage is a first-class caveat, not a footnote.** The rolling 12-quarter DD window needs
  ~4 years of clean history before a first signal; the monthly cross-section grows from ≈26 names
  (2012) to ≈40 (2024+), min 7. Terciles on a thin cross-section are noisy by construction; every
  number here should be read in that light.

## Survivorship — named on the Signal axis

The basket is **current survivors** (all still listed): a fixed roster of ~45 deep-history
non-financial names. It cannot include operating firms that were acquired or failed. For a
long-*high-quality*/short-*low-quality* book both legs are drawn from the same survivor pool, so
the first-order equity-survivorship tilt partly cancels; the residual risk is that
*quality-signal informativeness* itself is survivor-conditioned. We reason about the bias
direction explicitly rather than claiming it away, and never cite the survivor panel to certify
magnitude — moot here, since the return result is a null.

## Data sources

- **Net income, operating cash flow, total assets, receivables, inventory** — SEC EDGAR XBRL
  `companyconcept` API (`data.sec.gov`), 10-Q/10-K instant/duration facts, quarterly flows
  reconstructed from the YTD cumulative chain, de-duplicated on period end (earliest filing wins),
  keeping the filing date so the signal is strictly point-in-time. Cached under `_cache/aq_events.csv`.
- **Daily adjusted closes** — yfinance (no key), cached under `_cache/aq_prices.csv`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [231-sloan-accruals](../../231-sloan-accruals/) — the classic **Sloan (1996) accrual anomaly**:
  rank on the **level/sign of total accruals** scaled by assets (high accruals → low future
  returns). That grades *how much* of earnings is accrual; this study grades the **reliability**
  of accruals — the *volatility of the DD residual* — an orthogonal second-moment construct, not
  the accrual level.
- [522-percent-operating-accruals](../../522-percent-operating-accruals/) — **percent operating
  accruals** (accruals scaled by earnings rather than assets), still a *level* signal on the
  accrual component. This study uses no accrual level in the sort at all; the signal is the
  residual **standard deviation** from the DD cash-flow mapping.
- [539-cash-flow-volatility](../../539-cash-flow-volatility/) — sorts on the **volatility of
  operating cash flow** itself. Related in spirit (a second moment of the cash series) but
  distinct: DD accrual quality is the vol of the **accrual residual after regressing on cash
  flow** — it explicitly nets out the cash-flow variability that 539 sorts on, isolating the
  *estimation-error* part of accruals.
- [52-smoke-screen](../../52-smoke-screen/) — the earnings-management / "smoke screen" family
  (discretionary-accrual manipulation as a red flag). This study makes no discretionary-vs-normal
  split and no manipulation call; it measures the raw statistical *mapping quality* of accruals to
  cash, Dechow-Dichev style, and tests whether that mapping quality is priced.

None of the siblings rank on the **Dechow-Dichev residual volatility** — the reliability of the
accrual-to-cash mapping — which is this study's own axis.
