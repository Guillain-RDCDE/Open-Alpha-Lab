# References & literature map — Study 907 (Senior Loans vs High-Yield)

The seniority story sits at the intersection of leveraged-loan market structure, the credit
capital stack, and the empirical recovery-rate literature. Sources below; none of the claims
above rest on a single blog post.

## The instruments (what BKLN / SRLN / HYG / JNK actually hold)

- **Invesco Senior Loan ETF (BKLN)** — fund page & prospectus, tracks the Morningstar LSTA US
  Leveraged Loan 100 Index. Lists 2011-03-03. <https://www.invesco.com/us/en/etfs/senior-loan-etf-bkln.html>
- **SPDR Blackstone Senior Loan ETF (SRLN)** — actively managed senior-loan fund, lists
  2013-04-04. <https://www.ssga.com/us/en/institutional/etfs/spdr-blackstone-senior-loan-etf-srln>
- **iShares iBoxx $ High Yield Corporate Bond ETF (HYG)** and **SPDR Bloomberg High Yield Bond
  ETF (JNK)** — the two flagship USD high-yield *bond* ETFs. iShares: <https://www.ishares.com/us/products/239565/>
  · SSGA: <https://www.ssga.com/us/en/institutional/etfs/spdr-bloomberg-high-yield-bond-etf-jnk>
- **SPDR Bloomberg 1-3 Month T-Bill ETF (BIL)** — the cash / risk-free leg every Sharpe is
  measured excess of. <https://www.ssga.com/us/en/institutional/etfs/spdr-bloomberg-1-3-month-t-bill-etf-bil>

## The capital stack & recovery — why seniority *should* pay

- **Moody's Investors Service, Annual Default Study** (and the "Ultimate Recovery Database"):
  first-lien senior secured **loan** recoveries have historically run far above senior
  **unsecured bond** recoveries (~60–80% vs ~35–45%), the mechanical basis of the seniority
  claim. Moody's / S&P Global Ratings recovery studies (annual).
- **S&P Global Ratings / LCD (Leveraged Commentary & Data)** — leveraged-loan default and
  recovery statistics; the LSTA/Morningstar leveraged-loan index methodology.
- **Altman, E. & Kishore, V. (1996), "Almost Everything You Wanted to Know about Recoveries on
  Defaulted Bonds," *Financial Analysts Journal*** — seniority and security as the first-order
  determinants of recovery.
- **Schuermann, T. (2004), "What Do We Know About Loss Given Default?"** (Wharton FIC WP) — LGD
  is driven by seniority/collateral; the survey behind the "senior secured recovers more" fact.

## Floating rate, duration, and the liquidity catch

- **Leveraged loans float** off SOFR/LIBOR, so their price carries little interest-rate
  duration — the reason the loan sleeve barely fell in the 2022 rate shock (this study's stress
  table) and the subject of the sibling **Study 340 (Bank-Loans)** below.
- **Financial Stability Board (2019), "Vulnerabilities associated with leveraged loans and
  collateralised loan obligations"** — the liquidity-mismatch warning: daily-liquid loan
  *funds* holding an OTC, slow-settling asset can gap in a run, exactly the 2020 episode.
- **Federal Reserve / IMF Global Financial Stability Report (2020)** — leveraged-loan and
  loan-fund liquidity during the March 2020 dash-for-cash; loan-fund NAV discounts and forced
  selling. The empirical basis for "seniority is a solvency feature a liquidity crisis inverts."

## Method (inference & costing)

- **Newey, W. & West, K. (1987), "A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix," *Econometrica* 55** — the HAC *t* on the
  return spread.
- **Politis, D. & Romano, J. (1994), "The Stationary Bootstrap," *JASA* 89** — the circular
  block bootstrap behind the excess-Sharpe advantage CI.
- **Lo, A. (2002), "The Statistics of Sharpe Ratios," *Financial Analysts Journal*** — why a
  Sharpe (and a Sharpe *difference*) needs a standard error before you believe it; the ~0.24
  sampling SD that makes this study's advantage indistinguishable from zero.

## Sibling studies on this desk (dedup)

- **[Study 340 — Bank-Loans](../../340-bank-loans/)** — loans vs **rates** (BKLN's near-zero
  duration; the risk moves from duration to credit). *This* study is loans vs **high-yield**:
  a seniority race *within* the credit box, not a rate-protection test.
- **[Study 115 — Credit-Spreads](../../115-credit-spreads/)** — the level/timing of the HY
  credit spread as a signal; here we compare two *instruments* at similar carry, not the spread.
- **[Study 796 — Corporate-Bond-Low-Risk](../../796-corporate-bond-low-risk/)** — a
  low-risk/quality tilt *inside* corporate bonds; adjacent but a cross-section, not a
  loans-vs-HY sleeve race.
- **[Study 832 — High-Yield-Credit-Momentum](../../832-high-yield-credit-momentum/)** — a
  *timing* signal on HY; orthogonal to this static seniority comparison.

*Data via yfinance (public, total-return closes). Not investment advice — research & education.*
