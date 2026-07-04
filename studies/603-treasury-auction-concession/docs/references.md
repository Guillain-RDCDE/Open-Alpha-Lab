# References & literature map — Study 603 (Treasury Auction Concession)

## The claim under test

- **The folklore.** Every rates desk "knows" the pattern: the market **cheapens into** a big
  10Y/30Y auction (dealers and fast money push yields up to build a concession) and **richens
  after** the supply is absorbed. It shows up in sell-side previews weekly ("the market needs a
  concession into tomorrow's 30Y").
- **The formalisation.** Dong Lou, Hongjun Yan & Jinfan Zhang, *Anticipated and Repeated Shocks
  in Liquid Markets* (2013, **Review of Financial Studies** 26(8), 1891–1912). Secondary-market
  yields of Treasuries **rise in the days before** auctions of the same maturity and **fall
  back after** — a V-shaped price pattern around perfectly anticipated supply, which they read
  as limited dealer risk-bearing capacity plus slow-moving capital. They report the effect in
  amounts on the order of a few bps per auction cycle — the same order we find.
- **Dealer inventory mechanics.** Michael J. Fleming & Joshua V. Rosenberg, *How Do Treasury
  Dealers Manage Their Positions?* (2008, FRB New York Staff Report 299): primary dealers absorb
  auction supply and shed it over subsequent weeks, being compensated via price moves — the
  microfoundation of the concession.
- **Auction cycles and returns.** Additional context: Beetsma, Giuliodori, de Jong & Widijanto
  (2016, *JIMF*) on yield-cycle effects around euro-area auctions; Sigaux (2018, ECB WP) on
  pre-auction price declines; Forest (2012) on US auction announcement effects.

## Data

- **Auction records (official, complete).** U.S. Treasury FiscalData API,
  [`/v1/accounting/od/auctions_query`](https://fiscaldata.treasury.gov/datasets/treasury-securities-auctions-data/)
  — the feed behind [TreasuryDirect's auction query](https://www.treasurydirect.gov/auctions/auction-query/):
  every marketable auction since 1979 with auction date, offering amount, reopening flag,
  bid-to-cover, stop-out yield. We keep every **10-Year Note** and **30-Year Bond** auction by
  `original_security_term` (so reopenings — e.g. a 9-Year-10-Month — count as the supply events
  they are). No survivorship: this is the full official record.
- **Yields.** CBOE 10-Year (^TNX) and 30-Year (^TYX) constant-maturity yield indices, daily
  closes in percent, via [yfinance](https://github.com/ranaroussi/yfinance) (1962/1977 →).
  Constant-maturity indices, not a survivor panel.
- **Tradability tape.** TLT (iShares 20+ Year Treasury ETF) **auto-adjusted = total-return**
  closes and the 13-week T-bill rate ^IRX (the cash leg), via yfinance, 2002 →.

## Method citations (the desk's shared kit)

- **Newey & West (1987)**, *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix* — the HAC t on the event-dummy regression
  (overlapping 10Y/30Y refunding-week windows induce serial correlation by construction).
- **Welch (1947)**, *The generalization of "Student's" problem* — the big-vs-small size split
  and the (supporting-only) pre-window vs ordinary-week comparison.
- Event-window convention: pre-window **ends on the auction-day close** (results at 1 pm ET are
  inside that close); post-window is the five sessions after. One execution lag on the TLT leg:
  the auction calendar is public weeks ahead, entry at the auction-day close, first return
  earned A→A+1.

## Sibling studies on this bench (different plumbing — the dedup map)

- [382-treasury-basis-trade](../../382-treasury-basis-trade/) — the **cash-futures basis**
  carry trade (financing/delivery plumbing), not auction-cycle price pressure.
- [383-sofr-repo-stress](../../383-sofr-repo-stress/) — **repo-rate stress spikes** (money-market
  plumbing), not the duration concession around bond supply.
- [380-curve-roll-down](../../380-curve-roll-down/) — carry/roll-down along the curve, a static
  yield-curve property; this study is about the *event-time* V around scheduled supply.
