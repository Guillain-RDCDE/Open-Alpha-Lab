# References & literature map — Study 338 (Preferred-Stocks)

## The claim under test

**Preferred shares** — and the ETFs that hold them (PFF, PGX, PFFD, FPE) — are sold as a
*third asset class*: **bond-like safety with an equity-like yield.** The pitch is that a
preferred pays a fixed, fat coupon (~6%), sits **senior to common equity** in the capital
structure, and so delivers high income with "much less risk than stocks." Brokerage and
fund-company marketing routinely files preferreds under *fixed income* / *income & safety*.

- iShares' own positioning for **PFF** (*iShares Preferred and Income Securities ETF*,
  launched 2007-03-30) and Global X / Invesco analogues describe preferreds as a
  high-income, lower-volatility complement to equities.
- The retail framing ("get bond-like income with a juicier yield") is ubiquitous in
  income-investing media and dividend-focused communities.

## Why the steelman is almost coherent

- **Seniority is real.** Preferred dividends rank ahead of common dividends and, in
  liquidation, preferred holders are paid before common holders. On paper that *is* less
  risk than common equity.
- **The coupon is real and high.** Preferreds genuinely yield more than investment-grade
  bonds, because they bundle credit, subordination, and call/extension risk.

## Why it fails *as stated* ("bond-like safety")

- **Preferreds are deeply subordinated, perpetual, callable, and rate-sensitive** — a
  hybrid that behaves like equity in stress and like a bond in calm. The literature on
  hybrid/junior capital treats preferreds as carrying both credit-spread and equity-tail
  exposure (e.g. work on bank capital and contingent/junior securities; Standard & Poor's
  and Moody's hybrid-equity-credit methodologies that assign preferreds partial *equity*
  content for exactly this reason).
- **Heavy financial-sector concentration.** Most preferred issuance is by banks and
  insurers, so a preferred index is a leveraged bet on financial-sector health — precisely
  the sector that blows up in a crisis (2008). This is documented in the composition of PFF
  and peer indices.
- **The 2008 record is the canonical counter-example.** Preferred ETFs and indices fell
  ~60–70% in the GFC, alongside or worse than common equity, while Treasuries rallied —
  the opposite of the "safety" the label implies.

## Method lineage

- **Univariate / downside beta.** Conditioning beta on down-market days follows the
  downside-risk tradition (Bawa & Lindenberg 1977; Ang, Chen & Xing, *Downside Risk*, RFS
  2006) — the relevant measure when the claim is about behaviour *in the crash*.
- **Newey–West HAC standard errors** for the mean of an autocorrelated influence series
  (here, the OLS-beta influence function): Newey & West (1987), Econometrica.
- **Circular block bootstrap** for a CI on a beta *difference* — block resampling preserves
  volatility clustering and cross-asset co-movement that i.i.d. resampling destroys
  (Politis & Romano, 1994).
- **Total-return adjustment.** For an income instrument whose return is mostly coupon, the
  fair series is dividend-and-split adjusted (`yfinance auto_adjust=True`); a price-only
  series would understate PFF's return and is *not* used.

## Data sources used

- **PFF** (iShares Preferred & Income Securities ETF), **SPY** (equity), **IEF** (7-10y
  Treasuries), daily, **total-return adjusted** via `quantlab.data` (yfinance
  `auto_adjust=True`); SPY/IEF from the shared cross-asset cache, PFF cached study-local.
  PFF lists **2007-03-30**, which bounds the joint window honestly — stated as a decision,
  not buried.

## Related desk studies

- [Study 97 — Balancing-Act](../../97-balancing-act/) — the fixed 60/40 stock/bond blend.
  **Distinct**: 97 is an *allocation* race; 338 is a single-instrument *identity* test (is
  PFF a bond or a stock?).
- [Study 69 — Safe-Haven](../../69-safe-haven/) — does gold hedge crashes? Same crash-
  co-movement lens, different instrument. **Distinct**: 338 tests an instrument *marketed
  as* a bond, not a hedge.
- [Study 152 — Inflation-Hedge](../../152-inflation-hedge/) and
  [Study 207 — REITs-Diversifier](../../207-reits-diversifier/) — other "is this asset what
  the brochure says?" teardowns.
