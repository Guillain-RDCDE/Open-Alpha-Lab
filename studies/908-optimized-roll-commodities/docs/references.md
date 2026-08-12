# References & literature map — Study 908 (Optimized-Roll Commodities)

## The claim under test

- **The contango tax.** A long-only commodity index is a stack of futures that must be
  **rolled** before expiry — sell the expiring contract, buy a later one. When the curve is in
  **contango** (later contracts more expensive), that roll locks in a loss each cycle: the
  negative **roll yield** that dominates a fully-collateralised commodity index's return over
  time. Gorton & Rouwenhorst (2006, *Facts and Fantasies about Commodity Futures*, FAJ) and
  Erb & Harvey (2006, *The Strategic and Tactical Value of Commodity Futures*, FAJ) established
  that the term-structure (roll) component, not spot appreciation, drives the cross-section and
  the time series of index returns — so *how* you roll is first-order.
- **Optimized ("second-generation") roll indices.** Rather than roll naively into the front
  month, an optimized index picks the contract along the curve that minimizes carry cost (or
  maximizes backwardation), and the most aggressive versions screen the whole basket for the
  commodities currently in backwardation. The marketing claim: this **dodges the contango tax**
  and delivers a higher risk-adjusted return — a structural, mechanical edge. This study asks
  whether the packaged wrappers actually deliver it, excess of cash, over a full sample.

## The funds

- **USCI — United States Commodity Index Fund** (uscffunds.com): tracks the **SummerHaven
  Dynamic Commodity Index Total Return**. From a 27-commodity universe it selects the **14**
  with the strongest backwardation and 12-month price momentum, equal-weights them, and holds
  each in an **optimized (cheapest-to-hold) contract**. The canonical optimized-roll wrapper.
  Inception **2010-08-10** — it gates the common sample.
- **DBC — Invesco DB Commodity Index Tracking Fund** (invesco.com): a broad 14-commodity index
  using DB's **"Optimum Yield"** roll rule — within the next 13 months it selects the contract
  that maximizes implied roll yield (backwardation) or minimizes the loss (contango). *Semi-
  optimized*; the honest middle case and the study's primary benchmark. Expense ratio 0.85 %.
- **GSG — iShares S&P GSCI Commodity-Indexed Trust** (ishares.com): tracks the S&P GSCI,
  **world-production-weighted** (historically ~60-70 % energy) and rolling **naively into the
  front month** — the cleanest *front-month* comparator. Expense ratio 0.48 %.
- **DJP — iPath Bloomberg Commodity Index Total Return ETN** (ipathetn.barclays): tracks the
  Bloomberg Commodity Index, broad and diversification-capped, on the standard front-month
  Bloomberg roll schedule — a second front-month comparator with lighter energy weight.
- **PDBC — Invesco Optimum Yield Diversified Commodity Strategy No K-1 ETF**: the actively-
  managed, 1099 (no K-1) cousin of DBC, same **Optimum Yield** roll. Inception 2014-11 — a
  corroborating optimized roller on the shorter window.
- **BIL — SPDR Bloomberg 1-3 Month T-Bill ETF**: the **cash leg**. A collateralised commodity
  index bundles a full T-bill yield on top of spot + roll; that collateral return is identical
  across every wrapper and is not a roll edge, so every leg is measured **excess of BIL**.

## Why excess-of-cash, and why the sub-era cut is decisive

- **Collateral yield is not alpha.** Fully-collateralised commodity total return ≈ spot return +
  roll yield + **T-bill yield** on the collateral. At 2023-26 short rates the collateral leg is
  ~5 %/yr — large, and shared across USCI, DBC, GSG, DJP alike. Racing on *total* return would
  let a rate regime masquerade as a roll edge; subtracting BIL from both sides isolates the
  spot + roll difference, which is the only thing the claim is about.
- **Regime dependence.** The roll edge is not a constant premium — it is large when curves are in
  contango and can reverse when they flip to backwardation or when the optimized screen
  concentrates into the wrong commodities. A 16-year tape that spans essentially two commodity
  regimes cannot pin down a constant Sharpe advantage; the honest test is whether the edge holds
  *across* sub-eras, which is why the era cut carries the verdict.

## Method citations

- **Newey & West (1987)**, *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix*, Econometrica — the HAC *t* on the monthly
  return difference (6 Bartlett lags).
- **Politis & Romano (1994)**, *The Stationary Bootstrap*, JASA — the circular block bootstrap
  behind the paired 95 % CI on the Sharpe advantage (block length 6 months).
- **Lo (2002)**, *The Statistics of Sharpe Ratios*, FAJ — the standard-error and small-sample
  caveats on comparing Sharpe ratios (why a +0.10-+0.20 Sharpe gap on ~190 months is not a
  distinguishable-from-zero result).

## Related desk studies (dedup)

- [35-contango](../../35-contango/) — **times** the futures curve: a signal that moves a
  commodity position to cash or flips it based on whether the term structure is in contango or
  backwardation. **This study times nothing** — it compares two *always-invested* index
  wrappers (naive-roll vs optimized-roll) head to head.
- [794-commodity-carry](../../794-commodity-carry/) — a **cross-sectional** long-short of
  individual commodities sorted on carry / roll yield. Here we buy whole **packaged indices**
  and ask whether the optimized wrapper's structural roll rule pays a higher excess-of-cash
  Sharpe — a product-level race, not a factor sort.
- [661-uso-roll-decay](../../661-uso-roll-decay/) — the roll **decay** of a single-commodity
  front-month vehicle (USO, crude oil). This study is a **broad multi-commodity** index race
  and the optimized wrapper is the protagonist, not the victim.
- [226-crude-seasonality](../../226-crude-seasonality/) — a **calendar** effect in a single
  commodity; unrelated claim, no overlap.
