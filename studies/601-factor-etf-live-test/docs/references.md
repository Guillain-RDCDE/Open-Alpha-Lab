# References — Study 601 (Factor ETFs — Live Test)

## The claim's source

The 2011–2013 iShares "factor investing for everyone" launch wave — the promise that the
academic factor premia could be bought in a 0.15%/yr one-ticket wrapper:

- **iShares MSCI USA Min Vol Factor ETF (USMV)** — fund page & prospectus (launched
  2011-10-18; "reduce risk" is the stated objective).
  <https://www.ishares.com/us/products/239695/>
- **iShares MSCI USA Momentum Factor ETF (MTUM)** — launched 2013-04-16.
  <https://www.ishares.com/us/products/251614/>
- **iShares MSCI USA Value Factor ETF (VLUE)** — launched 2013-04-16.
  <https://www.ishares.com/us/products/251616/>
- **iShares MSCI USA Quality Factor ETF (QUAL)** — launched 2013-07-16.
  <https://www.ishares.com/us/products/256101/>
- **BlackRock, *Factor investing* marketing hub** — the retailized version of the pitch.
  <https://www.blackrock.com/us/individual/investment-ideas/what-is-factor-investing>

## Key papers

- **Ang, A., Hodrick, R., Xing, Y., Zhang, X. (2006), "The Cross-Section of Volatility and
  Expected Returns",** *Journal of Finance* 61(1) — the low-vol anomaly USMV wraps.
  <https://doi.org/10.1111/j.1540-6261.2006.00836.x>
- **Jegadeesh, N., Titman, S. (1993), "Returns to Buying Winners and Selling Losers",**
  *Journal of Finance* 48(1) — the 12-1 momentum construction our WML proxy copies.
  <https://doi.org/10.1111/j.1540-6261.1993.tb04702.x>
- **Fama, E., French, K. (1993), "Common Risk Factors in the Returns on Stocks and Bonds",**
  *JFE* 33(1) — HML, the value factor VLUE wraps. <https://doi.org/10.1016/0304-405X(93)90023-5>
- **Asness, C., Frazzini, A., Pedersen, L.H. (2019), "Quality Minus Junk",** *Review of
  Accounting Studies* 24 — the quality factor QUAL wraps.
  <https://doi.org/10.1007/s11142-018-9470-2>
- **Israel, R., Jiang, S., Ross, A. (2017), "Craftsmanship Alpha: An Application to Style
  Investing",** *JPM* — why implementation choices make live factor funds diverge from paper
  factors. <https://doi.org/10.3905/jpm.2018.44.2.023>
- **McLean, R.D., Pontiff, J. (2016), "Does Academic Research Destroy Stock Return
  Predictability?",** *Journal of Finance* 71(1) — post-publication factor decay, the prior
  for the alpha-delivery axis. <https://doi.org/10.1111/jofi.12365>

## Desk siblings (dedup guard)

- [**330-low-volatility-anomaly**](../../330-low-volatility-anomaly/) — the **academic
  cross-section** version of USMV's claim (SPLV/SPHB spread): Signal Weak, Tradability
  Fragile. This study does NOT re-litigate the anomaly; the unit under test is the **live
  iShares product** — did the wrapper deliver what its label promised, net of its own fee?
- [**242-quality-minus-junk**](../../242-quality-minus-junk/) — the academic QMJ factor
  itself. Here QUAL is audited as a shipped product against an independent-provider quality
  proxy (SPHQ), not as a long-short paper factor.

## Data sources

- **yfinance** (public, no key) — daily auto-adjusted (total-return) closes: SPY, USMV, MTUM,
  VLUE, QUAL; IWD/IWF (Russell 1000 Value/Growth — the value spread); SPHQ (Invesco S&P 500
  Quality — the independent quality proxy; index switch in 2016 flagged in results); the nine
  1998 SPDR sectors (XLB…XLY — raw material for the 12-1 sector WML proxy); ^IRX 13-week
  T-bill discount yield as the risk-free leg. <https://github.com/ranaroussi/yfinance>
- Method citations shared by the desk: Newey-West (1987) HAC errors; Welch (1947) unequal-
  variance *t*; Efron/Künsch moving-block bootstrap; permutation placebo (Fisher 1935).
