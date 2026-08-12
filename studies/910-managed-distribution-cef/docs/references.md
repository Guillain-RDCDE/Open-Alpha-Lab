# References & literature map — Study 910 (Managed-Distribution CEF)

## The claim under test

- **The closed-end-fund "double discount" folklore.** A closed-end fund (CEF) issues a fixed
  share count and trades on an exchange, so its market price can diverge from the net asset value
  (NAV) it holds — often at a **persistent discount**. The buyer's pitch: you buy a dollar of
  assets for ~90 cents (the *discount pull*) *and* collect a fat **managed distribution** (often
  8–14 % of NAV, paid on a fixed schedule) — a structural double-carry a passive index fund can't
  offer. The story is a staple of income-investing newsletters and CEF-screener services.
- **Managed distribution policies.** Under an SEC exemptive order (the "managed distribution
  plan", Investment Company Act §19(a)/§19(b) and Rule 19b-1), a CEF may pay a level distribution
  even when it exceeds net investment income and realised gains — the excess is **return of
  capital (ROC)**. "Destructive ROC" (paying out principal, eroding NAV) vs "constructive ROC"
  (pass-through of unrealised gains or of the discount) is the central diligence question; the
  §19(a) notice discloses the split. See SEC, *Investor Bulletin: Closed-End Fund Distributions*
  (sec.gov/investor).
- **Why the sceptic invokes the mREIT lesson.** Most of these CEFs are **leveraged** (~20–40 %
  effective leverage via preferred shares or repo, financed at short rates). When short rates
  rise, the financing cost climbs and leveraged NAVs fall faster — precisely the mechanism that
  made [611-mreit-carry](../../611-mreit-carry/) a Mirage. The honest test is therefore whether
  the *total* return (distribution reinvested) beats the asset class **risk-adjusted**, not
  whether the headline yield is large.

## Academic literature on the CEF discount

- **Lee, Shleifer & Thaler (1991)**, *Investor Sentiment and the Closed-End Fund Puzzle*, Journal
  of Finance 46(1) — the canonical treatment: discounts are wide, persistent, mean-reverting, and
  co-move (the "closed-end fund puzzle"), consistent with a noise-trader / sentiment risk premium.
- **Malkiel (1977)**, *The Valuation of Closed-End Investment-Company Shares*, JF 32(3) — the
  original discount-anomaly documentation and the arithmetic of buying assets below NAV.
- **Cherkes, Sagi & Stanton (2009)**, *A Liquidity-Based Theory of Closed-End Funds*, Review of
  Financial Studies — a rational account: the discount compensates for the illiquidity the wrapper
  packages, so the "free" discount is partly a fee for a service.
- **Pontiff (1996)**, *Costly Arbitrage: Evidence from Closed-End Funds*, QJE — why the discount
  persists (arbitrage is costly), and why a discount does not translate mechanically into excess
  return for the holder.

## The funds tested (fund pages / prospectuses, 2026)

- **PCEF** — Invesco CEF Income Composite ETF: a passive ETF holding ~130 taxable-income CEFs
  (the S-Network Composite Closed-End Fund Index); the diversified "CEF-of-CEFs" you buy in one
  click (invesco.com). Inception 2010-02.
- **PDI** — PIMCO Dynamic Income Fund: leveraged multi-sector bond CEF, one of the largest managed
  distributions in the space (~13 %/yr), frequently at a **premium** to NAV (pimco.com).
- **UTF** — Cohen & Steers Infrastructure Fund: leveraged listed-infrastructure equity CEF
  (cohenandsteers.com).
- **BST** — BlackRock Science and Technology Trust: option-overwrite (covered-call) tech-equity
  CEF with a managed distribution (blackrock.com); inception 2014-10 (the basket's binding start).
- **RQI** — Cohen & Steers Quality Income Realty Fund: leveraged real-estate (REIT) CEF — the
  clearest analogue to the mREIT-carry cautionary tale (cohenandsteers.com).
- **SPY** — the broad equity benchmark (the asset class you'd otherwise hold); **BIL** — SPDR
  1-3 Month T-Bill ETF, the cash / risk-free leg every Sharpe here is measured against.

## Data & method

- **Prices** — yfinance daily **total-return** closes (`auto_adjust=True`, so distributions are
  reinvested); cached once under `_cache/mdc_prices.csv`. We observe *market-price* total return,
  **not** the NAV / discount series — the discount cross-section itself is
  [367-closed-end-fund-discount](../../367-closed-end-fund-discount/)'s subject.
- **Newey & West (1987)**, *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix*, Econometrica — the HAC *t*'s on monthly means and
  on the excess-vs-excess CAPM (6 Bartlett lags), since CEF returns are serially correlated through
  discount mean-reversion and leverage-roll timing.
- **Politis & Romano (1994)**, *The Stationary Bootstrap*, JASA — the moving-block bootstrap behind
  the Sharpe confidence interval.

## Related desk studies (dedup)

- [367-closed-end-fund-discount](../../367-closed-end-fund-discount/) — the **cross-sectional
  discount signal** (buy the widest-discount CEFs, short the narrowest). **This study is
  different**: no discount cross-section is formed (yfinance gives price, not NAV); we test the
  **buyer's total-return bottom line** on a fixed basket of persistent-discount, big-distribution
  CEFs vs the asset class and cash.
- [611-mreit-carry](../../611-mreit-carry/) — the sibling **leverage-financed carry trap**
  (mortgage REITs pay a huge dividend but the levered book erodes book value). This study asks
  whether managed-distribution CEFs are the same trap in a different wrapper — and finds the payout
  is *real* (unlike the worst mREITs) but delivers no risk-adjusted edge over the asset class.
- [342-bdc-yield](../../342-bdc-yield/) — business-development companies: another high-yield,
  leveraged, NAV-vs-price closed wrapper; the **credit-carry cousin** on the private-lending side.
- [616-muni-cef-tax-loss](../../616-muni-cef-tax-loss/) — the **tax-driven** angle on muni CEFs
  (year-end tax-loss selling of discount CEFs), a seasonal trade, not a hold-the-payout carry.
