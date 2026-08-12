# References & literature map — Study 858 (Margin ÷ Inventory Divergence)

## The claim under test

- **The thesis.** In the fundamental-signals tradition of **Lev & Thiagarajan (1993)** and,
  most directly, **Abarbanell & Bushee (1997, 1998)**, a set of hand-built accounting signals
  predict future earnings and returns. Two of their signals are an **inventory** signal (the
  change in inventory relative to the change in sales — inventory *outrunning* sales is bad news)
  and a **gross-margin** signal (the change in gross margin relative to the change in sales). The
  folk version we test sharpens the pair into a single *contradiction* detector: a firm whose
  **gross margin is rising while inventory grows faster than sales** is telling two incompatible
  stories. Either the fat margin is unsustainable — it will have to be discounted to clear the
  stock that is piling up — or the swollen inventory is about to be written down. Coherence
  (margin up, inventory in line with sales) is the good picture; incoherence is the bad one.
- **The encoding.** We collapse the two Abarbanell–Bushee signals into
  **`divergence = (Δ gross-margin%) − (inventory-growth − sales-growth)`**, so a clean,
  coherent name scores **high** (the long) and a contradictory name scores **low** (the short).
  Ranking the cross-section on `divergence` and going long-top / short-bottom is the strong,
  tradable form of the claim.
- **The academic anchor.** Abarbanell & Bushee, "Fundamental Analysis, Future Earnings, and
  Stock Prices" (*Journal of Accounting Research*, 1997) and "Abnormal Returns to a Fundamental
  Analysis Strategy" (*The Accounting Review*, 1998); Lev & Thiagarajan, "Fundamental
  Information Analysis" (*JAR*, 1993). The inventory piece connects to **Thomas & Zhang (2002,
  RAST)**, "Inventory Changes and Future Returns," and the margin/earnings-quality piece to the
  accruals literature (**Sloan 1996**). These are among the oldest "fundamentals predict
  returns" results, and among the most heavily data-mined — a natural desk teardown.
- **The open question we test.** On a small, honestly-thin panel of large US inventory-carrying
  filers, does the divergence signal (a) **earn a forward return spread** (the mispricing
  claim) and (b) **lead the fundamentals** — predict next year's gross-margin change (the
  accounting mechanism) — once you rank strictly on point-in-time filed values, hold with one
  execution lag, and charge realistic long-short costs plus borrow?

## What we measure, and the honesty rails

- **Signal, point-in-time.** For each period end E, `gross_margin = (Rev−Cost)/Rev`,
  `d_gross_margin = GM(E) − GM(E−1yr)`, `inv_growth = Inv(E)/Inv(E−1yr) − 1`,
  `sales_growth = Rev(E)/Rev(E−1yr) − 1`, `inv_sales_gap = inv_growth − sales_growth`, and
  `divergence = d_gross_margin − inv_sales_gap`. Every input is known only at the **10-Q/10-K
  filing date** (`filed`, the latest of the revenue/cost/inventory filings), never the period
  end. A classic-Abarbanell variant that ranks on **`−inv_sales_gap` alone** (long low
  inventory-vs-sales growth) is carried as a robustness cut.
- **Primary test — calendar-time long-short (Newey-West).** Each month-end, rank the names that
  carry a fresh signal into terciles (the panel is too thin for quintiles), long the top / short
  the bottom equal-weight, earn the **next** month's return (one execution lag). The decisive
  statistic is the **Newey-West (HAC, Bartlett) t** of the monthly long-short series — the
  autocorrelation-robust bar `REAL` is written against. A one-sample t and a monthly hit-rate
  accompany it.
- **Cross-check — pooled event drift.** Bucket all (ticker, filing) events by the signal,
  measure top-minus-bottom forward drift over ≈1m/1q/2q horizons, with a one-sample t and a
  **label-shuffle placebo** (permute signals, re-form random terciles), plus the tercile
  monotonicity picture.
- **Third axis — does it actually lead the fundamentals?** A pooled OLS of **next year's**
  gross-margin change on the current divergence (slope, t, R², correlation) plus the
  future-margin spread between the top and bottom divergence terciles. This is the *mechanism*
  check: the contradiction can be real (low-divergence names really do see a later margin
  markdown) even if the stock does not move. The pooled t is read as suggestive — filings
  cluster by quarter, so it is not a calendar-robust HAC statistic.
- **Costs & borrow.** The tradability timer charges one-way cost × NAV × monthly turnover on
  **both** legs and makes the short leg pay an annualised borrow — the standard desk long-short
  friction model.
- **Coverage is a first-class caveat, not a footnote.** The quarterly (60–100-day span) filter
  drops the fiscal-Q4 figure disclosed only as a full-year span in the 10-K, and several names
  report Revenues / CostOfRevenue under different tags across the sample, so the per-name series
  has gaps and the monthly cross-section is uneven. Terciles on a thin cross-section are noisy
  by construction; every number here should be read in that light.

## Survivorship — named on the Signal axis

The basket is **current survivors** (all still listed): a fixed roster of ~46 inventory-carrying
US names — retailers, manufacturers, consumer-staples, apparel, hardware/semis, autos. It
cannot include the retailers and OEMs that were acquired or went bankrupt after an inventory
glut and a margin collapse — *exactly* the tail an inventory-contradiction short is supposed to
catch. That biases the short leg toward names that survived their contradictions, which if
anything works **against** finding the effect; we reason about the direction explicitly rather
than claiming it away, and never cite the survivor panel to certify magnitude.

## Data sources

- **Revenue, cost of revenue, inventory** — SEC EDGAR XBRL `companyconcept` API
  (`data.sec.gov`), 10-Q/10-K instant/duration facts, de-duplicated on period end (earliest
  filing wins), keeping the filing date so the signal is strictly point-in-time. Concept
  fallback ladders handle the tag salad (`Revenues` / `SalesRevenueNet` /
  `RevenueFromContractWithCustomerExcludingAssessedTax`; `CostOfRevenue` / `CostOfGoodsSold` /
  `CostOfGoodsAndServicesSold`; `InventoryNet`). Cached under `_cache/mid_events.csv`.
- **Daily adjusted closes** — yfinance (no key), cached under `_cache/mid_prices.csv`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [529-inventory-growth](../529-inventory-growth/) — ranks names on the **level of inventory
  growth** alone (Thomas–Zhang). This study does not rank on inventory growth; it ranks on the
  **contradiction** between the margin trend and the inventory-versus-sales gap — a two-signal
  divergence, not a one-signal level.
- [854-cash-conversion-cycle](../854-cash-conversion-cycle/) — the **working-capital cycle**
  (days inventory + receivables − payables), a liquidity/operating-efficiency construct. Our
  signal touches inventory but pairs it with the **gross-margin trend**, an income-statement
  quality signal, not a cash-cycle length.
- [122-gross-profitability](../122-gross-profitability/) — Novy-Marx **gross profits ÷ assets**,
  a *level* of profitability scaled by assets. This study uses the **change** in gross margin
  and only as one leg of a contradiction with inventory — not a profitability level.
- [231-sloan-accruals](../231-sloan-accruals/) — Sloan's **total-accruals** earnings-quality
  signal. Inventory build is one accrual, but the accruals anomaly is the aggregate
  balance-sheet accrual; this study isolates the specific **margin-vs-inventory-vs-sales**
  triangle of Abarbanell–Bushee.

None of the siblings rank on the **(ΔGross-margin) − (inventory-growth − sales-growth)
divergence** itself — this study's own axis.
