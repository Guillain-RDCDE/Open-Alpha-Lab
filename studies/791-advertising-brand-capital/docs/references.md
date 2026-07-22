# References & literature map — Study 791 (Advertising-Brand-Capital)

## The claim under test

- **The believers' version.** Advertising is expensed under GAAP, but the *brand awareness*
  and customer loyalty it buys is a durable intangible asset — **brand capital** — that never
  appears on the balance sheet. If the market under-values that hidden stock, firms that
  advertise heavily (high **AdvertisingExpense / Sales**) should earn a **forward return
  premium** as the mispricing corrects. It is the advertising-specific cousin of the broader
  "intangibles are mispriced" thesis.
- **The academic anchors.**
  - **Belo, Lin & Vitorino (2014), *Brand Capital and Firm Value*, Review of Economic
    Dynamics.** Build a brand-capital stock by perpetual-inventory accumulation of advertising
    spend; show it carries a risk/return signature in the cross-section and helps explain firm
    value. The canonical "advertising builds a priced intangible" reference.
  - **Chan, Lakonishok & Sougiannis (2001), *The Stock Market Valuation of Research and
    Development Expenditures*, Journal of Finance.** The intangibles-mispricing template: they
    study R&D but explicitly examine **advertising** too, finding the market's valuation of
    intangible-intensive firms is where mispricing predictability concentrates.
  - **Eisfeldt & Papanikolaou (2013), *Organization Capital and the Cross-Section of Expected
    Returns*, Journal of Finance** — the sibling "SG&A builds organization capital, and it is
    priced" result; advertising is a component of the same intangible-capital family.
- **The open question we test.** On a clean, point-in-time basket of firms that *actually file*
  the advertising line, does a simple advertising-intensity sort (long heavy advertisers, short
  light) earn a spread that clears the desk's `t >= 2` bar — or is the "brand premium" a
  literature result that a transparent tape cannot certify?

## What we measure, and the honesty rails

- **The signal.** `adv_sales = AdvertisingExpense(FY Y-1) / Revenue(FY Y-1)`, a pure
  spending-intensity characteristic — deliberately the **sales** denominator (a spending
  intensity), distinct from the *market-value* denominator that Chan-Lakonishok-Sougiannis find
  is the sharper mispricing signal for R&D (that denominator contrast is study 525's axis).
- **Portfolio sort.** Monthly rank; long the top tertile (heavy advertisers), short the bottom
  (light advertisers); equal-weight; monthly rebalance.
- **One execution lag, exact.** The signal at month-end *t* selects the book entered next month
  and held through *t+1* — one `shift`. On top of it, a **one-year reporting lag** on the
  accounting (the sort at any month in year Y uses the last fiscal year ending on or before
  Y-1), so no un-filed fundamental leaks into the ranking.
- **HAC inference.** Newey-West t-stat of the monthly long-short spread is the Signal-axis
  test. `REAL` needs **t >= 2 on this tape AND** survival of a label-shuffle placebo; the
  literature support alone reads `WEAK` (METHODOLOGY → *the inference bar*).
- **Placebo.** Permute the signal labels across names (same per-name values, wrong names),
  rebuild the long-short 400 times; the real spread must sit in the tail.
- **Costs one-way x NAV, shorts pay borrow.** `cost_bps` one-way turnover per rebalance (small
  here — advertising intensity is a slow, annual characteristic) plus an annual **borrow** fee
  on the short leg. Gross is labelled gross, net is labelled net.

## The coverage caveat (the honest scope, stated in the open)

- **Most firms omit the advertising line.** After ASU 2014-09 it stopped being a required
  disclosure, so the vast majority of US public companies file **no** `AdvertisingExpense`
  concept at all. Even inside this hand-picked *consumer* basket, some large advertisers
  (McDonald's, Costco, Estée Lauder, Booking, Hilton) never file it and are excluded by
  construction. So the tested "universe" is **the set of firms that advertise enough to
  disclose it** — a selected slice, not the market. This is stated on the Signal axis.
- **Concept heterogeneity.** A handful of names file the broader
  `MarketingAndAdvertisingExpense` tag (marketing + advertising bundled) rather than pure
  `AdvertisingExpense`; this mildly *over*-states advertising for those names, which we flag.
- **Survivorship.** The basket is current-membership projected back to 2010; acquired/delisted
  consumer names and names that dropped the disclosure are absent. Both legs are survivors, so
  the tilt is largely common to both legs of the long/short — reasoned about in writing on the
  Signal axis, not just buried in Tradability.

## Data sources

- **Fundamentals** — SEC EDGAR **companyconcept** (`AdvertisingExpense` /
  `MarketingAndAdvertisingExpense`, and `Revenues` / `RevenueFromContractWithCustomer…` /
  `SalesRevenueNet`), 10-K full-year facts only, as-first-reported (earliest `filed` per fiscal
  year). Public, no key. Cached under `_cache/` (`adv.parquet`, `rev.parquet`).
- **Prices / returns** — yfinance monthly total return (auto-adjusted close), cached
  (`returns.parquet`), 2010 → the last complete month, plus the SPY benchmark.
- Every headline number is pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [525-r-and-d-intensity](../525-r-and-d-intensity/) — the **R&D** intangible
  (Chan-Lakonishok-Sougiannis). Same intangible-mispricing family and same sort machinery, but
  the intangible is *research*, and its decisive axis is the **market-value vs sales**
  denominator contrast. This study is the **advertising** intangible on the **sales**
  denominator — a different characteristic (a Coca-Cola advertises heavily but does ~zero R&D;
  a semiconductor firm is the reverse).
- [526-intangible-value](../526-intangible-value/) — an **intangible-adjusted book value**
  (capitalise R&D + a share of SG&A into a corrected book-to-market). That is a *value* signal
  built by adding intangibles back to book equity; this study never touches book value — it
  ranks on a single, raw income-statement ratio (advertising / sales).
- [400-patent-intensity](../400-patent-intensity/) — **patents** (an innovation-output
  intangible) scaled by size. Same "hidden intangible is priced" thesis, a different intangible
  (granted patents, not advertising dollars) and a different data source (patent counts, not the
  10-K advertising line).

None of the siblings sort on the **advertising** line specifically — the brand-capital proxy —
which is this study's own axis.
