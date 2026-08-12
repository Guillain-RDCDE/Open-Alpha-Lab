# References — Study 900 (Quality-Income)

## The claim's source

The retail dividend-ETF debate: high **yield** chases value traps (the fattest yields
often mark payers about to cut), while **quality**-dividend screens (durable, growing
payers) were sold as the antidote. The four products we race:

- **Schwab U.S. Dividend Equity ETF (SCHD)** — tracks the Dow Jones U.S. Dividend 100,
  which screens for 10-year dividend consistency, cash-flow-to-debt, ROE, dividend
  growth and yield (a *quality + growth + yield* blend). Inception 2011-10-20.
  <https://www.schwabassetmanagement.com/products/schd>
- **ProShares S&P 500 Dividend Aristocrats ETF (NOBL)** — S&P 500 members with **25+
  consecutive years** of dividend increases (the aristocrat durability screen).
  Inception 2013-10-09. <https://www.proshares.com/our-etfs/strategic/nobl>
- **Invesco S&P 500 High Dividend Low Volatility ETF (SPHD)** — takes the 75
  highest-yielding S&P 500 names, then keeps the 50 least volatile (a *yield-first*
  screen). Inception 2012-10-18. <https://www.invesco.com/us/financial-products/etfs/product-detail?audienceType=Investor&ticker=SPHD>
- **Vanguard High Dividend Yield ETF (VYM)** — market-cap-weighted above-median-yield
  U.S. stocks (a broad *yield* cut). Inception 2006-11-10.
  <https://investor.vanguard.com/investment-products/etfs/profile/vym>
- **SPDR S&P 500 ETF (SPY)** — the plain cap-weight benchmark; **BIL** (SPDR Bloomberg
  1-3 Month T-Bill ETF) — the realized cash leg. <https://www.ssga.com/us/en/individual/etfs/spdr-sp-500-etf-trust-spy>

## Key papers / evidence

- **Asness, C., Frazzini, A., Pedersen, L.H. (2019), "Quality Minus Junk",** *Review of
  Accounting Studies* 24 — why *quality* (profitable, growing, safe payout) earns a
  premium the raw-yield screen misses. <https://doi.org/10.1007/s11142-018-9470-2>
- **Novy-Marx, R. (2013), "The Other Side of Value: The Gross Profitability Premium",**
  *JFE* 108(1) — profitability/quality as a distinct, priced characteristic.
  <https://doi.org/10.1016/j.jfineco.2013.01.003>
- **Arnott, R., Asness, C. (2003), "Surprise! Higher Dividends = Higher Earnings
  Growth",** *Financial Analysts Journal* 59(1) — high payout ratios need not signal
  weak growth; the "value trap" is about *unsustainable* yield, not yield per se.
  <https://doi.org/10.2469/faj.v59.n1.2504>
- **Baker, M., Bradley, B., Wurgler, J. (2011), "Benchmarks as Limits to Arbitrage:
  Understanding the Low-Volatility Anomaly",** *FAJ* 67(1) — the low-vol screen SPHD
  bolts onto its yield sort. <https://doi.org/10.2469/faj.v67.n1.4>
- **Blitz, D., van Vliet, P. (2007), "The Volatility Effect",** *JPM* 34(1) — a prior for
  why SPHD's low-vol leg can decouple from the raw-yield leg in stress.
  <https://doi.org/10.3905/jpm.2007.698039>

## Desk siblings (dedup guard)

- [**206-dividend-aristocrats**](../../206-dividend-aristocrats/) — grades the *academic
  aristocrat signal* (consecutive-raise screen) as a long-short characteristic. Study 900
  does not re-litigate that signal; it **races two live dividend product-sleeves** on
  excess-of-cash Sharpe and drawdown.
- [**233-shareholder-yield**](../../233-shareholder-yield/) — total shareholder yield
  (dividends + buybacks + debt paydown) as a factor; here we test *dividend* screens only,
  as shipped ETFs, quality-vs-raw-yield.
- [**57-yield-trap**](../../57-yield-trap/) — the yield-trap phenomenon itself (does the
  very-highest-yield decile underperform?); study 900 is the *product* framing — does a
  quality-dividend sleeve you can actually buy beat a high-yield sleeve you can buy?
- [**601-factor-etf-live-test**](../../601-factor-etf-live-test/) — the live-ETF audit
  template this study copies (excess-of-cash Sharpe races, NW *t*, block-bootstrap CIs,
  planted-parameter synthetic control); 601 audits the iShares *factor* wrappers, 900 the
  *dividend* wrappers.

## Data sources

- **yfinance** (public, no key) — daily auto-adjusted (total-return) closes for SCHD,
  NOBL, VYM, SPHD, SPY and BIL. <https://github.com/ranaroussi/yfinance>
- Method citations shared by the desk: Newey-West (1987) HAC standard errors; Efron /
  Künsch / Politis-Romano moving-block bootstrap for the Sharpe-gap CI; Lo (2002) on the
  standard error of the Sharpe ratio.
