# References — Study 902 (Multi-Factor Composite)

## The claim's source

The practitioner pitch that a **blend** of factor sleeves dominates any single factor —
"factors take turns working, so diversify across them" — and the retail multi-factor products
that package it:

- **iShares / BlackRock, *Factor investing* & multi-factor hub** — the retailized "combine the
  factors" pitch behind products like LRGF (iShares U.S. Equity Factor ETF).
  <https://www.blackrock.com/us/individual/investment-ideas/what-is-factor-investing>
- The five single-factor iShares ETFs blended here (all 0.08–0.15%/yr expense):
  **VLUE** value <https://www.ishares.com/us/products/251616/>,
  **QUAL** quality <https://www.ishares.com/us/products/256101/>,
  **MTUM** momentum <https://www.ishares.com/us/products/251614/>,
  **USMV** min-vol <https://www.ishares.com/us/products/239695/>,
  **SIZE** size <https://www.ishares.com/us/products/251082/>.

## Key papers

- **Fama, E., French, K. (1993), "Common Risk Factors in the Returns on Stocks and Bonds",**
  *JFE* 33(1) — size (SMB) and value (HML), two of the sleeves.
  <https://doi.org/10.1016/0304-405X(93)90023-5>
- **Jegadeesh, N., Titman, S. (1993), "Returns to Buying Winners and Selling Losers",**
  *Journal of Finance* 48(1) — momentum, the MTUM sleeve.
  <https://doi.org/10.1111/j.1540-6261.1993.tb04702.x>
- **Asness, C., Frazzini, A., Pedersen, L.H. (2019), "Quality Minus Junk",** *Review of
  Accounting Studies* 24 — quality, the QUAL sleeve.
  <https://doi.org/10.1007/s11142-018-9470-2>
- **Ang, A., Hodrick, R., Xing, Y., Zhang, X. (2006), "The Cross-Section of Volatility and
  Expected Returns",** *Journal of Finance* 61(1) — the low-vol anomaly USMV wraps.
  <https://doi.org/10.1111/j.1540-6261.2006.00836.x>
- **Asness, C., Ilmanen, A., Israel, R., Moskowitz, T. (2015), "Investing with Style",** *JPM*
  — the canonical case for *combining* value/momentum/carry/defensive into one multi-style
  portfolio (the diversification pitch this study tests live).
  <https://doi.org/10.3905/jpm.2015.42.2.015>
- **Grinold, R., Kahn, R. (1999), *Active Portfolio Management*** — the fundamental law: the
  information ratio of a blend of imperfectly-correlated signals rises with breadth (why the
  blend beats the *average* sleeve — the half of the pitch that survives here).
- **McLean, R.D., Pontiff, J. (2016), "Does Academic Research Destroy Stock Return
  Predictability?",** *Journal of Finance* 71(1) — post-publication factor decay, the prior for
  why the *live* factor premia (and thus the blend's edge over the market) may have thinned.
  <https://doi.org/10.1111/jofi.12365>

## Desk siblings (dedup guard)

- [**601-factor-etf-live-test**](../../601-factor-etf-live-test/) — the closest sibling: it
  audited **each** iShares factor ETF's *exposure delivery* vs SPY (does USMV really carry a
  low-vol profile, does MTUM load on momentum?). This study is the **portfolio question** — a
  single **combined equal-weight sleeve** raced against SPY on excess-of-cash Sharpe, net of
  the blend's own rebalancing turnover. Different unit (a blend, not five separate funds),
  different test (a Sharpe race + diversification decomposition, not per-fund exposure
  regressions).
- [**638-value-momentum-everywhere**](../../638-value-momentum-everywhere/) — the *academic*
  value+momentum long-short combination (AQR "Value and Momentum Everywhere"), a paper factor
  cross-asset; here the sleeve is five **long-only live ETFs**, not paper long-shorts.
- [**401-signal-stacking**](../../401-signal-stacking/) — the general research-method demo of
  *stacking* imperfectly-correlated signals; this is its live single-instance realization on
  the five shipped factor products.
- [**242-quality-minus-junk**](../../242-quality-minus-junk/) — the QMJ academic factor itself;
  QUAL enters here only as one long-only member of the blend, not as a long-short factor.

## Data sources

- **yfinance** (public, no key) — daily auto-adjusted (total-return) closes: VLUE, QUAL, MTUM,
  USMV, SIZE (the five single-factor sleeves); SPY (benchmark); BIL (SPDR 1-3-month T-bill ETF,
  the tradable cash/risk-free leg). <https://github.com/ranaroussi/yfinance>
- Method citations shared by the desk: Newey-West (1987) HAC errors; Efron/Künsch moving-block
  bootstrap; Politis-Romano (1994) circular block bootstrap (quantlab `sharpe_ci_bootstrap`).
