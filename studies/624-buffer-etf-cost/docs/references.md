# References & literature map — Study 624 (Buffer ETFs — the Cost of Comfort)

## The claim under test

- **The product pitch.** Defined-outcome ("buffer") ETFs promise a stated downside buffer
  against the reference asset's **price** return over a 12-month outcome period, in exchange
  for a cap on upside participation, packaged in FLEX options. Issuer term sheets:
  - Innovator ETFs, *U.S. Equity Power Buffer ETF series* (PJAN / PAPR / PJUL / POCT):
    15% buffer on SPY's price return, annual reset on the first trading day of the name
    month, expense ratio 0.79%/yr — https://www.innovatoretfs.com/define/
  - First Trust / FT Vest, *FT Vest Laddered Buffer ETF (BUFR)*: equal-weight ladder of 12
    monthly FT Vest Buffer ETFs (~10% buffer each), expense ratio 0.95%/yr —
    https://www.ftportfolios.com/retail/etf/etfsummary.aspx?Ticker=BUFR
  (Terms hardcoded in [`data.py`](../buffer_etf_cost/data.py) with source comments,
  fetched 2026-07-03.)
- **The critique we test.** "You pay the cap plus a fat fee for insurance you could build
  cheaper with index + T-bills" — the standard advisor-press critique of the category, e.g.
  Morningstar's coverage of defined-outcome ETFs (Armour/Johnson, *Buffer ETFs: More Costly
  Than Beneficial?*, morningstar.com) and Cliff Asness's broader "why buy expensive
  vol-laundering?" line (AQR, *Rebalancing and Buffer ETFs*, aqr.com perspectives).

## Key papers & background

- **Israelov, R. & Lu, D. (2019/2022), *The Hidden Cost of Buffer Funds* (NDVR / SSRN):**
  argues defined-outcome payoffs are replicable with index + cash and that the wrapper's
  fee and structuring costs make them dominated ex ante. The direct academic form of the
  claim we test on the tape.
- **Israelov, R. & Nielsen, L. (2015), *Covered Calls Uncovered*, FAJ** — the option-selling
  decomposition (equity beta + short vol) that underlies why capped structures are mostly
  low-beta equity. The desk's covered-call sibling ([337](../337-covered-call-etf/)) leans on
  the same lineage.
- **Bhansali, V. et al. on tail-risk hedging cost (e.g., *Tail Risk Hedging*, 2014)** — why
  systematic protection tends to be fairly-to-richly priced; the buffer's put-spread-collar
  is the retail packaging of that trade.
- **Newey, W. & West, K. (1987)** — HAC standard errors used for every gap *t* here.
- **Frazzini, Israel & Moskowitz (2018), *Trading Costs*** — the gross-vs-net discipline
  behind charging explicit rebalancing costs on the DIY mix.

## What we measure, and the honesty choices

- **Delivery on the stated reference.** Outcome-period terms are written on SPY's **price**
  return (dividends excluded), so the delivery check pairs fund **total** return with SPY
  **price** return between month-end closes bracketing each reset. Performance races,
  by contrast, are always **total-return vs total-return** — both labeled everywhere.
- **The "dumb mix it replaces".** w·SPY + (1−w)·BIL with w = the fund's full-sample monthly
  beta (a descriptive estimate, flagged as such), monthly rebalanced with explicit one-way
  costs on the traded NAV; a fixed-w grid (0.45–0.70) shows the verdict does not hinge on
  the in-sample beta. Both legs fully funded — excess-vs-excess by construction.
- **One documented lag.** The mix trades only at month-end closes to weights known in
  advance; no same-bar information.
- **Survivor slice.** BUFR + the four oldest Power Buffer vintages are the category's
  successful flagships (later small series have closed). Named on the Signal axis; the bias
  flatters the funds, i.e. it works *against* the cost claim we end up confirming.
- **Synthetic controls** ([`data.synthetic_world`](../buffer_etf_cost/data.py),
  [`synthetic_outcomes`](../buffer_etf_cost/data.py)) plant a tunable structuring drag
  (gap detector) and a known cap/buffer (delivery checker) — machinery proofs only, never
  cited as market evidence.

## Data sources

- **yfinance** daily closes (auto-adjusted and raw), tickers BUFR, PJAN, PAPR, PJUL, POCT,
  SPY, BIL, 2018-08-01 → 2026-06-30, cached under `_cache/bec_adj.csv` / `_cache/bec_raw.csv`.
  All headline numbers pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (dedup map)

- [337-covered-call-etf](../337-covered-call-etf/) — the **income** option wrapper (sell the
  upside for a distribution): that study is about yield illusion and NAV erosion. This study
  is the **defined-outcome** wrapper (buy a buffer, accept a cap): the question is delivery
  and the price of insurance — a different structure and a different verdict (the buffers
  delivered and priced fairly; the buy-writes leaked).
- [99-safety-net](../99-safety-net/) — DIY drawdown insurance via trailing stops (timing,
  not options); [370-zero-dte-options](../370-zero-dte-options/) — option *selling* flow.
  Neither tests a packaged defined-outcome product.
