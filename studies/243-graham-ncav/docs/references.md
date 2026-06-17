# References — Study 243 (Graham NCAV)

## Primary sources

1. **Graham, B. & Dodd, D. (1934).** *Security Analysis.* McGraw-Hill. First
   codification of the NCAV criterion: buy stocks priced below net current asset
   value (current assets minus all liabilities). Graham later set the two-thirds
   rule as the margin of safety.

2. **Graham, B. (1949).** *The Intelligent Investor.* Harper & Brothers.
   Chapter 7 discusses "Bargain Issues" — stocks priced at two-thirds of NCAV
   or less. Graham estimated such stocks provided ~15%/yr excess returns in
   his era.

## Academic evidence on NCAV

3. **Oppenheimer, H.R. (1986).** "Ben Graham's Net Current Asset Values: A
   Performance Update." *Financial Analysts Journal*, 42(6), 40–47.
   Documents substantial positive returns (average +29.4%/yr) for NCAV stocks
   over 1970-1983. Small-cap, illiquid universe; data pre-dates modern quant.

4. **Vu, J.D. (1988).** "An Empirical Analysis of Ben Graham's Net Current
   Asset Value Rule." *Financial Review*, 23(2), 215–225.
   Confirms Oppenheimer's findings for 1977-1984; notes declining opportunity
   set even then.

5. **Bildersee, J., Cheh, J. & Zutshi, A. (1993).** "The Performance of
   Japanese Common Stocks in Relation to Their Net Current Asset Values."
   *Japan and the World Economy*, 5(2), 197–215.
   NCAV screen applied to Japanese market in the 1970s-1980s: positive returns
   suggest the anomaly existed in other developed markets.

6. **Lauterbach, B. & Vu, J.D. (1993).** "Ben Graham's Net Current Asset
   Value Rule Revisited: The Size Effect." *Quarterly Journal of Business and
   Economics*, 32(1), 82–108.
   Size effect explains most of NCAV outperformance; controlling for size
   reduces the alpha significantly.

## Decline and irrelevance of NCAV in large caps

7. **Xiao, Y. & Arnold, G. (2008).** "Testing Benjamin Graham's Net Current
   Asset Value Strategy in London." *Journal of Investing*, 17(4), 11–19.
   UK evidence: NCAV stocks outperform 1981-2005, but the universe is tiny and
   composed of micro-cap, illiquid stocks. Not reproducible at scale.

8. **Carlisle, T. (2010).** *Quantitative Value.* Wiley. Chapter on NCAV
   discusses that genuine net-nets have largely disappeared from US large-cap
   indices by the 1990s. The screen requires micro/small-cap to find candidates.

9. **Greenwald, B. et al. (2001).** *Value Investing: From Graham to Buffett
   and Beyond.* Wiley. Documents that Warren Buffett himself moved away from
   NCAV-based investing ("cigar butt" investing) because the opportunity set
   exhausted itself and he needed larger, quality businesses.

## Why large-cap "net-nets" are not what Graham meant

10. **Fama, E. & French, K. (1992).** "The Cross-Section of Expected Stock
    Returns." *Journal of Finance*, 47(2), 427–465.
    Documents the value premium (P/B, not NCAV), concentrated in small-cap.
    In large-cap, value premia are smaller and less robust.

11. **Campbell, J., Hilscher, J. & Szilagyi, J. (2008).** "In Search of
    Distress Risk." *Journal of Finance*, 63(6), 2899–2939.
    Distressed firms do not earn higher returns in US large-cap — the "distress
    puzzle." Related to why cheap large-cap firms do not systematically outperform.

## Data sources

12. **EDGAR (SEC).** Balance sheet data (CurrentAssets, TotalLiabilities,
    WeightedAverageDilutedShares) from 10-K annual filings. Via the desk's
    shared EDGAR concept cache (_cache/_edgar_*.parquet).

13. **yfinance.** Monthly adjusted close prices for S&P 500 members; December
    month-end prices used for market-cap computation.
