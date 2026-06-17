# References & literature map -- Study 273 (Lego-Returns)

## The claim under test

**Dobrynskaya, V. & Kishilova, J. (2022).** "LEGO: The Toy of Smart Investors."
*Research in International Business and Finance*, 59, 101539.
The headline academic source. Using a large sample of retired sets traded on the
secondary market (BrickPicker/BrickLink) over 1987-2015, the authors estimate an
average price-only return of roughly 8-11%/yr -- higher than large stocks, bonds,
gold and many art categories over the same window -- with **near-zero correlation
to financial markets**. They are careful to flag the high cross-set dispersion,
the illiquidity, and that returns are pre-transaction-cost. The press telescoped
this into "LEGO beats the S&P 500", dropping the fees, storage, and selection bias.

## Why the "beats stocks" headline does not survive

- **Wrong benchmark.** A collectible pays no cash flow, so it must be compared to
  the **price-only** equity index, not total return. The S&P's ~2%/yr dividend is
  a real return the LEGO holder forgoes. Our comparison is price-only on both legs
  and reports the dividend gap separately.

- **Gross vs net.** Index appreciation is a *paper* return. Realising it means
  selling through a marketplace: eBay final-value + payment fees run ~12-13%,
  BrickLink ~3% plus shipping, photography and time. Amortised over a multi-year
  hold this is a large recurring drag. Dobrynskaya-Kishilova report gross returns.

- **Survivorship / selection bias.** A secondary-market "index" is built from sets
  that *kept trading and were tracked*. Sets that flopped, were liquidated below
  RRP, or vanished from resale are under-represented, biasing the index upward.
  Published LEGO-as-investment returns are therefore **upper bounds** -- named on
  the Signal axis of this study.

- **Liquidity & operations.** Each set is a single, non-fungible item. A
  market-beating basket is dozens of bulky boxes needing climate-controlled
  storage; selling 100 sets is 100 listings and shipments. None of this appears in
  an index return.

## Academic literature on collectibles as investments

- **Burton, B. J. & Jacobsen, J. P. (1999).** "Measuring Returns on Investments in
  Collectibles." *Journal of Economic Perspectives*, 13(4), 193-212. The canonical
  survey: collectibles (art, stamps, wine, antiques) historically underperform
  equities once you account for selection bias, transaction costs, insurance and
  storage; the headline returns are gross and survivorship-flattered.

- **Goetzmann, W. N. (1993).** "Accounting for Taste: Art and the Financial Markets
  over Three Centuries." *American Economic Review*, 83(5), 1370-1376. Art "indices"
  built from repeat-sales overstate returns because only works worth re-auctioning
  re-enter the sample -- the same selection mechanism that flatters a LEGO index.

- **Dimson, E. & Spaenjers, C. (2011).** "Ex Post: The Investment Performance of
  Collectible Stamps." *Journal of Financial Economics*, 100(2), 443-458. Real
  collectible returns are positive but modest, volatile, and below equities
  net of costs -- a useful template for reading LEGO claims sceptically.

## Method lineage

- **Newey-West HAC t-stat.** `hac_tstat` implements the standard heteroskedasticity-
  and autocorrelation-consistent variance estimator with Bartlett weights and the
  rule-of-thumb lag `floor(4*(n/100)^(2/9))`. On annual data the lag is ~1; annual
  excess returns have negligible serial correlation, so the HAC t is close to the
  plain t. The bar for a REAL signal is a **positive** mean excess with |t| >= 2.
- **CAPM beta.** OLS slope of LEGO annual returns on the market via
  `scipy.stats.linregress`; near-zero beta supports the diversification half of the
  claim while saying nothing about the return level.
- **Paired sign test.** `scipy.stats.binomtest` on the count of years LEGO beats the
  market, null = 0.5.
- **Cost model.** A single one-way resale friction amortised linearly over the
  holding period -- conservative (it ignores buy-side premium over RRP, storage and
  insurance).

## Data sources

- **S&P 500 daily price.** `_cache/^GSPC_split_only.parquet` (split-adjusted,
  PRICE-ONLY, no dividends), staged in the repo-level cache. December-to-December
  calendar-year price returns 1987-2024 (n=38).
- **LEGO secondary-market index.** Hardcoded in `data.py` as annual index levels
  (base 100 at year-end 1986), curated from the era-level returns in
  Dobrynskaya-Kishilova (2022) and extended to 2024 with conservative post-boom
  BrickLink/BrickEconomy aggregate trends. A stylised index, not a tick reconstruction.

## Related desk studies

- Alternative-asset / collectibles folklore that share the gross-vs-net and
  survivorship traps (wine, watches, trading cards, NFTs).
- The "uncorrelated diversifier" framing connects to the permanent-portfolio and
  60/40 allocation studies elsewhere in the lab.
