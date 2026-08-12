# References & literature map — Study 854 (Cash Conversion Cycle)

## The claim under test

- **The thesis.** The **Cash Conversion Cycle (CCC)** measures how many days a firm's cash is
  tied up in operations between paying its suppliers and collecting from its customers:
  `CCC = DSO + DIO − DPO`, where `DSO = AccountsReceivable / (Revenues / 365)` (days sales
  outstanding), `DIO = Inventory / (COGS / 365)` (days inventory outstanding) and
  `DPO = AccountsPayable / (COGS / 365)` (days payables outstanding). A firm that **shortens**
  its CCC — collecting faster, holding less inventory, stretching its payables float — frees
  working capital it can redeploy; a firm whose CCC is **rising** is a working-capital drag,
  bleeding cash into receivables and inventory faster than its payables can fund. The prediction
  is directional — **falling CCC good, rising CCC bad** — so the tradeable expression is
  **long the CCC-shorteners, short the CCC-bloaters**, sorted on the year-over-year change in CCC.
- **The academic / practitioner anchor.** The CCC construct is due to **Richards & Laughlin
  (1980, "A Cash Conversion Cycle Approach to Liquidity Analysis", *Financial Management*)**. A
  long operations-finance literature links a **shorter** CCC to higher profitability — e.g.
  **Deloof (2003, *Journal of Business Finance & Accounting*)**, **Shin & Soenen (1998,
  *Financial Practice and Education*)** and the annual **CFO / Hackett "Working Capital
  Scorecard"** — the practitioner tradition that treats freed working capital as a value driver.
  It is a cousin of the accruals literature (**Sloan 1996, *The Accounting Review***: receivables
  and inventory are accruals) and overlaps the working-capital pieces of the **Piotroski (2000)
  F-score**. CCC is, in effect, the *net* of two forensic red flags — rising receivables
  (Study 853) and rising inventory (Study 529) — offset by the payables float.
- **The open question we test.** On a small, honestly-thin panel of large US filers that report
  all five CCC ingredients on EDGAR, does the **year-over-year change in CCC** (a) **precede a
  better operating margin** (the "frees cash → out-earns" mechanism) and (b) **earn a forward
  return spread** in the claimed direction (shorteners minus bloaters), once you rank strictly on
  point-in-time filed values, hold with one execution lag, and charge realistic long-short costs
  plus borrow?

## What we measure, and the honesty rails

- **Signal, point-in-time.** `ccc_yoy_chg` = current-quarter CCC minus the same quarter a year
  earlier, in **days**, known only at the **10-Q/10-K filing date** (`filed`), never the period
  end. A unit-free percentage-change variant (`ccc_yoy_pct`) is carried as a robustness cut. The
  ranking column `ccc_score = −ccc_yoy_chg` is signed so that HIGH score = FALLING CCC = the
  attractive (long) side; a *wrong-sign* result surfaces as a *negative* long-short. Concepts:
  `AccountsReceivableNetCurrent`, `InventoryNet`, `AccountsPayableCurrent` (instants) plus the
  quarterly flows `Revenues` and `CostOfRevenue` (COGS fallback `CostOfGoodsAndServicesSold`),
  longest per-name history. Each quarter of a flow is annualised to a daily run-rate; the
  annualisation factor and seasonality cancel in the same-fiscal-quarter YoY change.
- **Primary test — calendar-time long-short (Newey-West).** Each month-end, rank the names that
  carry a fresh signal into terciles (the panel is too thin for quintiles), long the
  falling-CCC third / short the rising-CCC third equal-weight, earn the **next** month's return
  (one execution lag). The decisive statistic is the **Newey-West (HAC, Bartlett) t** of the
  monthly long-short series — the autocorrelation-robust bar `REAL` is written against
  (METHODOLOGY → *The inference bar*). A one-sample t and a Wilson-interval monthly hit-rate
  accompany it.
- **Cross-check — pooled event drift.** Bucket all (ticker, filing) events by the signed score,
  measure top-minus-bottom forward drift over ≈1m/1q/2q horizons, with a one-sample t and a
  **label-shuffle placebo** (permute signals, re-form random terciles).
- **Third axis — does shortening precede a better margin?** A pooled OLS of **next quarter's**
  gross-margin change on this quarter's CCC change (slope, t, R², correlation) plus the
  future-margin-change spread between the shortening and bloating terciles. The "frees cash →
  out-earns" story predicts a **negative** slope. This is the *mechanism* check; the pooled t is
  read as suggestive — filings cluster by quarter, so it is not a calendar-robust HAC statistic.
- **Costs & borrow.** The tradability timer charges one-way cost × NAV × monthly turnover on
  **both** legs and makes the short leg pay an annualised borrow — the standard desk long-short
  friction model.
- **Coverage is a first-class caveat, not a footnote.** The CCC needs *five* matched facts per
  quarter (three balances + two flows). Each has tagging quirks — COGS is `CostOfRevenue` for
  some filers and `CostOfGoodsAndServicesSold` for others, and ASC-606 shifted revenue tags in
  2018 — so the quarters where all five line up cleanly are fewer than the raw name count
  suggests, and thin early in the sample. Terciles on a thin cross-section are noisy by
  construction; every number here should be read in that light.

## Survivorship — named on the Signal axis

The basket is **current survivors** (all still listed): a fixed roster of ~46 large US filers
that carry genuine inventory and trade payables and report the concepts today. It cannot include
firms that were acquired or delisted. For a long-*shorteners*/short-*bloaters* signal both legs
are drawn from the same survivor pool, so the first-order equity-survivorship tilt partly
cancels; the residual risk is that *signal informativeness* itself is survivor-conditioned. We
therefore reason about the bias direction explicitly rather than claiming it away, and never cite
the survivor panel to certify magnitude.

## Data sources

- **Accounts receivable, inventory, accounts payable (current), quarterly revenue and COGS** —
  SEC EDGAR XBRL `companyconcept` API (`data.sec.gov`), 10-Q/10-K instant/duration facts,
  de-duplicated on period end (earliest filing wins), keeping the filing date so the signal is
  strictly point-in-time. Cached under `_cache/ccc_events.csv`. CIKs are resolved from the SEC
  company-ticker map.
- **Daily adjusted closes** — yfinance (no key), cached under `_cache/ccc_prices.csv`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [853-days-sales-outstanding](../853-days-sales-outstanding/) — the **receivables** leg alone
  (rising DSO as a channel-stuffing red flag). CCC nets DSO against *two more* legs (inventory
  and payables); a firm can have a rising DSO but a *falling* CCC if it stretches payables or
  clears inventory. This study ranks on the **whole cycle**, not one leg.
- [529-inventory-growth](../529-inventory-growth/) — **inventory growth**, the DIO leg's cousin
  (goods piling up unsold). CCC is inventory *days* netted against receivables and payables, not
  raw inventory growth.
- [153-net-operating-assets](../153-net-operating-assets/) — the **balance-sheet bloat**
  anomaly (Hirshleifer et al.): cumulative operating accruals scaled by assets. CCC is a
  *flow-scaled days* measure of one slice of working capital, and we rank on its **year-over-year
  change**, not the NOA *level* scaled by total assets.
- [524-operating-leverage](../524-operating-leverage/) — **operating leverage** (the fixed-cost
  structure driving earnings sensitivity). A cost-*structure* signal, unrelated to the
  working-capital *timing* the CCC measures.

None of the siblings rank on the **year-over-year change in the Cash Conversion Cycle** itself —
DSO + DIO − DPO as a single netted number — this study's own axis.
