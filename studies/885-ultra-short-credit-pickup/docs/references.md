# References — Study 885 (Ultra-Short Credit Pickup)

## The idea's source

- **The ultra-short / enhanced-cash sleeve.** The pitch, made by every issuer of these funds,
  is that short-maturity investment-grade credit earns a **spread over T-bills** for a small,
  well-diversified sliver of credit and duration risk — an "enhanced cash" or "cash-plus" carry
  that should improve reward-per-unit-of-risk versus pure bills.
  - **PIMCO, "Enhanced Short Maturity Active ETF (MINT)"** — the granddad of the sleeve
    (inception 2009-11), the issuer's statement of the short-IG-credit-over-cash rationale.
    https://www.pimco.com/en-us/investments/etf/enhanced-short-maturity-active-exchange-traded-fund
  - **J.P. Morgan Asset Management, "Ultra-Short Income ETF (JPST)"** (inception 2017-05) — an
    actively managed ~AA-/A ultra-short book (corporates + ABS, ~0.25–0.9y duration).
    https://am.jpmorgan.com/us/en/asset-management/adv/products/jpmorgan-ultra-short-income-etf-etf-shares-46641q837
  - **iShares (BlackRock), "Ultra Short-Term Bond Active ETF (ICSH)"** (inception 2013-12).
    https://www.ishares.com/us/products/258806/

## Why "spread over bills" is a real economic quantity

- **Duffie, D. & Singleton, K. (2003), *Credit Risk: Pricing, Measurement, and Management*.**
  The credit spread compensates expected loss plus a risk/liquidity premium — the structural
  reason ultra-short IG paper yields more than a matched-maturity T-bill.
- **Ang, A. (2014), *Asset Management: A Systematic Approach to Factor Investing*, ch. on fixed
  income.** Term and credit premia as compensated factors; also why a *thin* premium is easily
  swamped by its own drawdown timing and by short samples.
- **Newey, W. & West, K. (1987), "A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix", *Econometrica* 55(3).** The HAC covariance
  behind every *t* quoted here — essential because ultra-short-credit NAV marks are serially
  correlated, which inflates a naive Sharpe/t.
- **Lo, A. (2002), "The Statistics of Sharpe Ratios", *Financial Analysts Journal* 58(4).** Why
  Sharpe races are run excess-of-cash and why autocorrelation must be corrected before reading a
  Sharpe *t*.
- **Politis, D. & Romano, J. (1994), "The Stationary Bootstrap", *JASA* 89.** The circular block
  bootstrap behind the Sharpe confidence interval.

## Data sources

- **yfinance** (Yahoo! Finance, public, no key) — daily auto-adjusted (total-return, net-of-fee)
  closes: JPST, ICSH, MINT, BIL, SHV. https://pypi.org/project/yfinance/
- **JPST** — JPMorgan Ultra-Short Income ETF (inception 2017-05-17, ER 0.18%).
- **ICSH** — iShares Ultra Short-Term Bond Active ETF (inception 2013-12-11, ER 0.08%).
- **MINT** — PIMCO Enhanced Short Maturity Active ETF (inception 2009-11-16, ER 0.35%) — the
  longest live tape, used for the 16.6-year robustness cut.
- **BIL** — SPDR Bloomberg 1-3 Month T-Bill ETF (inception 2007-05-25) — the tradable cash /
  risk-free leg; every Sharpe race is excess-of-BIL.
- **SHV** — iShares Short Treasury Bond ETF (0-1y Treasuries, inception 2007-01-05) — the second
  cash benchmark, a hair more duration than BIL but ~zero credit.

## Sibling studies on this desk (distinct claims — the dedup map)

- [Study 340 — Bank-Loans](../../340-bank-loans/) tests **floating-rate high-yield loans** (BKLN)
  as a *duration-for-credit* risk swap — deep sub-IG credit, near-zero duration. This study is the
  opposite corner: *ultra-short IG* credit with a *sliver* of both credit and duration, measured
  as a near-cash pickup.
- [Study 338 — Preferred-Stocks](../../338-preferred-stocks/) tests **preferred equity** yield —
  long-duration, subordinated, equity-hybrid risk — not a cash substitute.
- [Study 577 — MBS-OAS-Signal](../../577-mbs-oas-signal/) tests **mortgage-backed spread (OAS) as
  a timing signal**, a when-to-hold-risk claim, not a static who-you-park-cash-in pickup.
- [Study 625 — Starting-Yield](../../625-starting-yield-bond-decade/) tests whether a bond fund's **starting
  yield mechanically predicts its forward return** — a within-fund identity across the whole
  maturity spectrum, not the cross-sectional credit-over-bills spread tested here.

## Shared method citations

- Repo-wide protocol: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar
  (HAC *t* ≥ 2 on the real tape, a bootstrap CI clear of zero, and sub-era robustness for `Real`),
  excess-vs-excess Sharpe races, one documented lag, synthetic controls as machinery proofs only.
- `quantlab.stats` / `quantlab.analytics` — `annualized_sharpe`, `sharpe_ci_bootstrap`,
  `mean_tstat_hac`, `sharpe_with_se` reused here for the Sharpe race and its confidence interval.
