# References & literature map — Study 621 (Share-Class Spreads)

## The claim under test

- **The conversion terms, from the source.** Warren E. Buffett, *Comparative Rights and
  Relative Prices of Berkshire Class A and Class B Stock* (Berkshire Hathaway memo, updated
  after the 2010 split) — <https://www.berkshirehathaway.com/compab.pdf>. Each Class A share is
  convertible **at the holder's option, at any time, into 1,500 Class B shares** (30 before the
  50:1 B split of 2010-01-21); **Class B is never convertible into Class A**. Buffett's own
  words: the B "can never sell for anything more than a tiny fraction above" 1/1500th of the A
  price — when it does, "arbitrage takes place" — but it **"can sell at a discount"** because
  nothing converts the other way. That memo *is* the claim we test.
- **The B-share issue.** Berkshire issued Class B in May 1996 (first trade 1996-05-09)
  explicitly to pre-empt unit trusts splitting the A; the 2010 50:1 B split accompanied the
  BNSF acquisition and B's entry into the S&P 500.
- **The Google split.** Alphabet distributed non-voting Class C (GOOG) to GOOGL (Class A,
  1 vote) holders on 2014-04-03. **No conversion bridge exists in either direction**, so the
  A−C spread floats free. The 2014 split agreement even included a (one-off, 2015) true-up
  payment to C holders because the spread had no anchor — the cleanest admission that the
  market knew there was no bound.

## Key papers

- Lamont & Thaler (2003), *Can the Market Add and Subtract? Mispricing in Tech Stock
  Carve-outs*, **JPE 111(2)** — the canonical "law of one price violated where arbitrage is
  blocked" study (Palm/3Com); our BRK/GOOG contrast is the same logic run on conversion rights.
- Schultz & Shive (2010), *Mispricing of Dual-Class Shares: Profit Opportunities, Arbitrage,
  and Trading*, **JFE 98(3)** — dual-class twin spreads mean-revert and most apparent profits
  sit inside the bid-ask; matches our fill-at-print collapse exactly.
- Zingales (1995), *What Determines the Value of Corporate Votes?*, **QJE 110(4)** and
  Nenova (2003), *The Value of Corporate Voting Rights and Control*, **JFE 68(3)** — voting
  premia are small in strong-governance regimes; the GOOGL premium's drift to *negative* adds
  the buyback-tilt twist (Alphabet repurchases concentrate in class C, supporting GOOG).
- Shleifer & Vishny (1997), *The Limits of Arbitrage*, **JF 52(1)** — why a bound enforced by
  a privileged arbitrage (A-holders minting B) coexists with a persistent discount no one can
  arb away.
- Newey & West (1987), *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix*, **Econometrica 55(3)** — the HAC t used on the
  heavily autocorrelated daily gaps.

## Method notes (what we did and why)

- **Split-adjusted parity.** On `auto_adjust=True` closes the 2010 50:1 B split folds the
  pre-2010 ratio of 30 into a constant **1,500**, so one parity covers the whole tape. Neither
  BRK class has ever paid a dividend; Google's 2024+ dividends are identical per share on both
  classes and cancel in the ratio. Returns are price-only on adjusted closes and both legs of
  every comparison use the same convention.
- **Close-print noise, named.** BRK-A trades a few hundred times a day; its 4 p.m. print is
  stale/wide relative to B's. Apparent sub-10-bps "premiums" are non-synchronous prints, so
  bound enforcement is judged at 50–100 bps. The same noise is exactly why the fill-at-print
  overlay shows a fake +4.3%/yr: the signal *is* the noise you pretend to trade at.
- **One execution lag.** Signal at close *t*, filled at close *t+1*, first return accrues
  *t+1 → t+2*. The fill-at-print variant (fill at close *t*) is reported only as the
  diagnostic that locates the mirage.
- **Costs.** One-way × NAV per leg; a class switch = 2 legs; the pairs short leg pays
  50 bps/yr borrow (both classes are easy borrow).
- **Synthetic control.** [`data.synthetic_pair`](../share_class_spreads/data.py) plants a
  tunable mean discount and a hard one-way bound; the null (no bound, zero discount) must not
  trigger the detector. A machinery proof only.

## Data sources

- **yfinance** daily split-adjusted closes: BRK-A, BRK-B (1996-05-09 →), GOOG, GOOGL
  (2014-04-03 →), cached under `_cache/scs_prices.csv`; as-of 2026-06-30, fingerprint
  `145603f5d518` ([`docs/results.md`](results.md)).
- Berkshire Hathaway, *compab.pdf* (conversion terms): <https://www.berkshirehathaway.com/compab.pdf>
- Alphabet 2014 class-C distribution & 2015 true-up: Google Inc. 8-K/10-K filings, SEC EDGAR
  (<https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001288776>).

## Related desk studies (dedup framing)

This study is **pure share-class arithmetic on a hard conversion right** — a *mechanical* bound
tested against its unbounded twin. It is unrelated to
[05-twin-spread](../05-twin-spread/) (statistical distance-pairs on co-moving tickers — no
conversion right, no parity, a purely empirical spread), and complementary to
[367-closed-end-fund-discount](../367-closed-end-fund-discount/) (NAV discounts with no
conversion at all), [618-gbtc-premium-cycle](../618-gbtc-premium-cycle/) (a wrapper premium
whose creation/redemption bridge was one-way then broken) and
[620-a-h-premium](../620-a-h-premium/) (same-company classes segmented by market access). Here
the bound is a *contractual option held by every A shareholder* — the only one of the family
the market must enforce daily.
