# References — Study 887 (High-Yield Muni Premium)

## The claim's source

- **The high-yield municipal credit premium.** High-yield munis (below investment grade or
  non-rated) pay a materially higher yield than investment-grade munis to compensate for default and
  — especially — liquidity risk. The premium is the marketing backbone of every HY-muni fund:
  VanEck, "High Yield Muni ETF (HYD)" fund materials describe the strategy as harvesting the
  incremental spread of the HY-muni segment over the broad muni market.
  https://www.vaneck.com/us/en/investments/high-yield-muni-etf-hyd/
- **Tax-equivalent yield.** Municipal-bond interest is exempt from federal income tax (and the 3.8%
  net-investment-income surtax). The standard tax-equivalent-yield identity, TEY = y / (1 − t), is the
  textbook comparison a taxable investor uses to weigh a muni against a taxable bond of comparable
  risk. MSRB, "Tax-Equivalent Yield" (Electronic Municipal Market Access / investor education).
  https://www.msrb.org/

## Key papers & evidence on the mechanism

- **Schwert, M. (2017), "Municipal Bond Liquidity and Default Risk"**, *Journal of Finance* 72(4):
  decomposes the muni spread into default vs liquidity components — the liquidity piece is large and
  is exactly what blows out in crises (the 2020/2022 episodes this study isolates).
  https://doi.org/10.1111/jofi.12511
- **Ang, A., Bhansali, V. & Xing, Y. (2010), "Taxes on Tax-Exempt Bonds"**, *Journal of Finance*
  65(2): the effective tax treatment and pricing of municipal bonds — why the after-tax comparison,
  not the pre-tax one, is the economically correct race for a taxable investor.
  https://doi.org/10.1111/j.1540-6261.2010.01554.x
- **Green, R. (2007), "Presidential Address: Issuers, Underwriter Syndicates, and Aftermarket
  Transparency"**, *Journal of Finance* 62(4): the muni market's segmentation and illiquidity — the
  structural reason HY-muni gaps hard when dealers step back.
- **Newey, W. & West, K. (1987), "A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix"**, *Econometrica* 55(3): the HAC covariance behind
  every *t* quoted here. https://doi.org/10.2307/1913610
- **Politis, D. & Romano, J. (1994), "The Stationary Bootstrap"**, *JASA* 89(428): the block
  bootstrap behind the mean confidence interval (serially dependent monthly returns).
- **Lo, A. (2002), "The Statistics of Sharpe Ratios"**, *Financial Analysts Journal* 58(4): why
  Sharpe races are run excess-vs-excess on a tradable risk-free.

## Data sources

- **yfinance** (Yahoo! Finance, public, no key) — daily auto-adjusted (total-return, net-of-fee)
  closes plus a price-only pull for the income decomposition. https://pypi.org/project/yfinance/
- **HYD** — VanEck High Yield Muni ETF (inception 2009-02-04, ER 0.35%); the HY-muni leg.
- **MUB** — iShares National Muni Bond ETF (inception 2007-09-07, ER 0.05%); the IG-muni benchmark.
- **TFI** — SPDR Nuveen Bloomberg Municipal Bond ETF (inception 2007-09, ER 0.23%); an alternative
  IG-muni control.
- **HYG** — iShares iBoxx $ High Yield Corporate Bond ETF (ER 0.49%); the *taxable* high-yield
  yardstick for the tax-equivalent comparison.
- **BIL** — SPDR Bloomberg 1-3 Month T-Bill ETF; the tradable risk-free proxy (excess-vs-excess races).

## Sibling studies on this desk (distinct claims — the dedup map)

- [Study 576 — Muni-Treasury-Ratio](../../576-muni-treasury-ratio/) tests the muni/Treasury *yield
  ratio* as a rich/cheap valuation-timing signal. This study holds no valuation view — it tests a
  *credit* spread (HY-muni over IG-muni) as a carried premium.
- [Study 616 — Muni-CEF-Tax-Loss](../../616-muni-cef-tax-loss/) tests a *seasonal tax-loss-selling*
  effect in muni closed-end funds. This study tests a structural credit-and-tax premium in liquid
  open-end muni ETFs, no calendar seasonality.
- [Study 115 — Credit-Spreads](../../115-credit-spreads/) tests whether *widening HY spreads predict
  equity returns* (a cross-asset timing claim). This study measures a *within-muni* credit premium
  you carry, not a timing signal.
- [Study 610 — Fallen-Angels-Premium](../../610-fallen-angels-premium/) is the closest cousin — a
  *within-credit selection* premium in taxable HY. Study 887 is its **tax-exempt** analogue and asks
  the extra question 610 does not: does the tax wrapper change the after-tax verdict?

## Shared method citations

- Repo-wide protocol: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (HAC *t* ≥ 2 on
  the real tape for `REAL`), excess-vs-excess races, one documented lag, bootstrap CIs, synthetic
  controls as machinery proofs only.
