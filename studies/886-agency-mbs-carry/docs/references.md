# References & literature map — Study 886 (Agency MBS Carry)

## The claim under test

- **Agency MBS carry a spread over duration-matched Treasuries.** A mortgage pass-through
  is a Treasury bond plus a **short prepayment option**: the homeowner can refinance (call
  the bond) when rates fall and drag his feet when they rise. The investor is short that
  option, which makes MBS **negatively convex**, and is paid an **option-adjusted spread
  (OAS)** — historically ~20-60 bp — as compensation. The "carry" story says: buy MBS,
  duration-hedge with Treasuries, and pocket that spread as a real, positive premium.
- **Negative convexity is the risk being paid.** Because the effective duration of an
  MBS *shortens* as rates fall (prepayments accelerate) and *extends* as rates rise, the
  realized rate beta of an MBS sits well below its static option-adjusted duration — and
  the position loses exactly when rates move sharply. The premium exists to compensate for
  that; the question this study asks is whether the *realized* premium, net of the convexity
  drag, is actually positive and robust on the live ETF tape.

## Method & data sources

- **ETFs (yfinance, `auto_adjust=True` total-return closes).**
  - **MBB** — iShares MBS ETF (agency pass-throughs), inception 2007-03; the longest MBS
    tape here (ishares.com).
  - **VMBS** — Vanguard Mortgage-Backed Securities ETF, inception 2009-11 (investor.vanguard.com);
    the corroborating MBS tape (misses the GFC).
  - **IEF** — iShares 7-10 Year Treasury Bond ETF (ishares.com): the duration-matched
    Treasury leg (~7.5y effective duration brackets MBB/VMBS's ~6y).
  - **AGG** — iShares Core US Aggregate Bond ETF: a broad investment-grade reference.
  - **BIL** — SPDR Bloomberg 1-3 Month T-Bill ETF (ssga.com): the cash / risk-free leg;
    every return is taken excess of BIL (the excess-vs-excess rail).
- **Effective durations** (fund pages, mid-2026): MBB ≈ 6.0y, VMBS ≈ 6.0y, IEF ≈ 7.5y —
  used for the static OAD-ratio hedge (`β = 6.0 / 7.5 = 0.80`) alongside the empirical
  (regression) hedge.
- **Newey & West (1987)**, *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix*, Econometrica — the HAC *t* on the carry
  mean (6 Bartlett lags; 3/12 sensitivity reported).
- **Politis & Romano (1994)**, *The Stationary Bootstrap*, JASA — the circular block
  bootstrap behind the 95% CI on the annualized carry mean.

## Literature

- **Boudoukh, Richardson, Stanton & Whitelaw (1997)**, *Pricing Mortgage-Backed Securities
  in a Multifactor HJM Framework*, RFS — MBS as Treasuries plus a short prepayment option;
  the OAS as the compensation for that embedded optionality.
- **Gabaix, Krishnamurthy & Vigneron (2007)**, *Limits of Arbitrage: Theory and Evidence
  from the Mortgage-Backed Securities Market*, JF — prepayment (negative-convexity) risk is
  priced; MBS OAS reflects a genuine risk premium, not just a free spread.
- **Diep, Eisfeldt & Richardson (2021)**, *The Cross Section of MBS Returns*, JF — documents
  that agency-MBS excess returns compensate for prepayment risk, with the premium
  concentrated in and varying with the rate environment (consistent with this study's
  regime dependence).
- **Duarte, Longstaff & Yu (2007)**, *Risk and Return in Fixed-Income Arbitrage*, RFS — the
  MBS carry/convexity trade as a fixed-income arbitrage strategy whose returns are eaten by
  the tail risk it is short.

## Related desk studies (dedup)

- [577-mbs-oas-signal](../../577-mbs-oas-signal/) — tests whether agency-MBS **OAS widening
  is a risk-off *signal*** (a cross-asset lead indicator). **This study is different**: we do
  not use OAS as a timing signal — we **harvest the carry** itself, the realized excess
  return of the MBS over a duration-matched Treasury, and ask if that premium is real and
  bankable.
- [340-bank-loans](../../340-bank-loans/) — the packaged-credit-carry cousin (leveraged-loan
  ETFs): a different embedded risk (credit / floating-rate) in a different wrapper.
- [796-corporate-bond-low-risk](../../796-corporate-bond-low-risk/) — the low-risk-anomaly
  angle in corporate bonds, not a duration-hedged carry.
- [581-term-premium](../../581-term-premium/) — the Treasury **term premium** (duration risk
  itself); here the Treasury leg is *hedged out* so what remains is the MBS-specific
  prepayment/convexity spread, not the term premium.
