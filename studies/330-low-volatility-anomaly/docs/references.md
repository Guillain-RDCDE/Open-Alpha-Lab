# References & literature map — Study 330 (Low-Volatility-Anomaly)

## The claim, at full strength

- **Baker, Bradley & Wurgler (2011)**, *"Benchmarks as Limits to Arbitrage: Understanding the
  Low-Volatility Anomaly."* *Financial Analysts Journal* 67(1). The canonical statement: low-vol
  and low-beta stocks earn *higher* risk-adjusted returns; benchmark-relative mandates deter the
  leveraged arbitrage that would close the gap.
- **Frazzini & Pedersen (2014)**, *"Betting Against Beta."* *Journal of Financial Economics*
  111(1). Leverage-constrained investors bid up high-beta assets, flattening the security-market
  line; a beta-neutral long-low/short-high book (BAB) earns a premium. The method behind this
  study's *beta-neutral* spread.
- **Ang, Hodrick, Xing & Zhang (2006)**, *"The Cross-Section of Volatility and Expected
  Returns."* *Journal of Finance* 61(1). High idiosyncratic-vol stocks subsequently earn *lower*
  returns — the idio-vol puzzle (this desk's [Study 54 — Static](../../54-static/)).
- **Blitz & van Vliet (2007)**, *"The Volatility Effect: Lower Risk Without Lower Return."*
  *Journal of Portfolio Management* 34(1). Global evidence that low-vol portfolios match market
  returns at lower risk — the higher-Sharpe claim.

## The product side (the ETFs we actually race)

- **Invesco S&P 500 Low Volatility ETF (SPLV)** — holds the 100 S&P 500 names with the lowest
  trailing realised volatility, vol-weighted; launched 2011-05.
- **Invesco S&P 500 High Beta ETF (SPHB)** — holds the 100 S&P 500 names with the highest
  trailing beta to the index; launched 2011-05. The literal "exciting" counterpart to SPLV.
- **S&P Dow Jones Indices**, methodology for the *S&P 500 Low Volatility* and *S&P 500 High Beta*
  indices — the rebalanced rules behind the two funds.

## Neighbours on this bench (the dedup map)

- **[Study 18 — Dull-Roar](../../18-dull-roar/)** — the academic low-vol anomaly on the S&P 500
  *cross-section* (decile sorts, the Frazzini–Pedersen BAB long-short). Study 330 is the
  *tradable ETF* embodiment of the same idea, not the cross-section.
- **[Study 58 — Bunker](../../58-bunker/)** — the min-vol ETF (USMV) vs the *market* (SPY).
  Study 330 instead races low-vol against its *high-beta opposite* (SPLV vs SPHB), the full
  boring-vs-exciting spread, not low-vol vs market.
- **[Study 54 — Static](../../54-static/)** — the *idiosyncratic*-vol (residual) puzzle. Study
  330 uses *total* realised vol / beta, the funds' actual selection rule.
- **[Study 43 — Free-Lunch](../../43-free-lunch/)** / **[238 — Betting-Against-Beta](../../238-betting-against-beta/)**
  — the betting-against-beta factor on single names. Study 330 is the off-the-shelf ETF version.

## Shared method

- **Newey & West (1987)** — heteroskedasticity- and autocorrelation-consistent (HAC) standard
  errors; the *t*-stat the inference bar requires.
- **Politis & Romano (1994)**, **Lahiri (1999)** — circular/stationary block bootstrap; preserves
  the short-run autocorrelation that i.i.d. resampling destroys (this study's CI on the spread).
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar, the
  excess-of-cash Sharpe rule, one execution lag, costs one-way × NAV with shorts paying borrow.
