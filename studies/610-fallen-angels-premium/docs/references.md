# References — Study 610 (Fallen-Angels-Premium)

## The claim's source

- **Ben Dor, A. & Xu, Z. (2011), "Fallen Angels: Characteristics, Performance, and
  Implications for Investors"**, *Journal of Fixed Income* 20(4).
  The canonical study: bonds downgraded from investment-grade to high-yield suffer forced
  selling ahead of/around the index ejection, become oversold, and outperform comparable
  original-issue high-yield bonds over the following months.
  https://www.pm-research.com/content/iijfixinc/20/4/33
- **ICE/BofA & Bloomberg fallen-angel index studies (index history since 1996-12-31).**
  The ICE US Fallen Angel High Yield Index (H0FA) and Bloomberg US HY Fallen Angel 3% Cap
  Index have outperformed broad US high-yield since inception — the marketing backbone of the
  claim. Index-era evidence is **context only** on this desk: it is graded through the live,
  fee-paying tape. https://indices.theice.com/
- **VanEck, "Fallen Angel Bonds: Upgrade Your High Yield"** — the issuer's own statement of
  the mechanism (forced sellers at downgrade → discounted entry → recovery + rising-star
  upgrades). https://www.vaneck.com/us/en/investments/fallen-angel-high-yield-bond-etf-angl/

## Key papers on the mechanism

- **Ellul, A., Jotikasthira, C. & Lundblad, C. (2011), "Regulatory Pressure and Fire Sales in
  the Corporate Bond Market"**, *Journal of Financial Economics* 101(3): insurance companies,
  bound by capital rules, sell downgraded bonds *because they must* — the cleanest
  identification of the forced-seller leg. https://doi.org/10.1016/j.jfineco.2011.03.020
- **Ambrose, B., Cai, N. & Helwege, J. (2008), "Forced Selling of Fallen Angels"**, *Journal
  of Fixed Income* 18(1): price pressure around IG→HY downgrades and the subsequent reversal.
- **Newey, W. & West, K. (1987)** — the HAC covariance behind every *t* quoted here.
- **Lo, A. (2002), "The Statistics of Sharpe Ratios"** — why Sharpe races are run
  excess-vs-excess.

## Data sources

- **yfinance** (Yahoo! Finance, public, no key) — daily auto-adjusted (total-return,
  net-of-fee) closes: ANGL, FALN, HYG, JNK, IEF, LQD, BIL. https://pypi.org/project/yfinance/
- **ANGL** — VanEck Fallen Angel High Yield Bond ETF (inception 2012-04-10, ER 0.35%).
- **FALN** — iShares Fallen Angels USD Bond ETF (inception 2016-06-14, ER 0.25%).
- **HYG / JNK** — iShares iBoxx $ HY Corporate Bond ETF (ER 0.49%) / SPDR Bloomberg High
  Yield Bond ETF — the broad high-yield benchmarks.
- **IEF / LQD / BIL** — 7-10y Treasuries (duration factor), IG credit (quality factor),
  1-3m T-bills (tradable risk-free).

## Sibling studies on this desk (distinct claims — the dedup map)

- [Study 115 — Credit-Spreads](../../115-credit-spreads/) tests whether *widening HY spreads
  predict equity returns* (a cross-asset timing claim). This study tests a *within-credit
  selection* premium — who you hold inside high-yield, not when you hold risk.
- [Study 340 — Bank-Loans](../../340-bank-loans/) tests whether *floating-rate loans are a
  safe bond substitute* (a duration-vs-credit risk-swap claim). This study holds the duration
  axis fixed (IEF control) and asks whether the *downgrade-driven entry price* pays.

## Shared method citations

- Repo-wide protocol: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar
  (HAC *t* ≥ 2 on the real tape for `REAL`), excess-vs-excess races, one documented lag,
  synthetic controls as machinery proofs only.
