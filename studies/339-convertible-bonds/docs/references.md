# References & literature map — Study 339 (Convertible-Bonds)

## The claim under test

A **convertible bond** is a corporate bond carrying an embedded option to convert into a
fixed number of the issuer's shares. The strong, sold-at-full-strength version is that this
gives you a **convex payoff** — *"equity upside with bond downside"*: when the stock rallies
the option dominates and you ride it up; when the stock falls the bond value (the "bond
floor") and the coupon cushion the loss. Drawn as a payoff diagram it is a hockey-stick — a
curve that bends *upward* (positive gamma).

- The convexity pitch is the standard sell-side framing for the asset class — e.g. SSGA's
  marketing for **CWB** (SPDR Bloomberg Convertible Securities ETF) and Calamos /
  Advent-style convertible literature describe converts as offering "equity-like upside
  participation with downside protection from the bond floor."
- The academic framing of a convertible as a bond + an equity call / warrant goes back to
  the contingent-claims valuation of Ingersoll (1977) and Brennan & Schwartz (1977,
  *Convertible bonds: valuation and optimal strategies for call and conversion*, JF).

## Why the steelman is almost coherent

- **A long option genuinely is convex.** A call's payoff is convex in the underlying
  (positive gamma), so a bond-plus-call *should*, in principle, capture more of the upside
  than the downside (Black & Scholes 1973; Merton 1973). At the **single-name** level a
  balanced (at-the-money) convertible can show real convexity.
- **The "bond floor" exists for the issue, in isolation.** Absent default, the straight-bond
  value bounds a single convertible's price from below.

## Why it is likely to fail *as stated* for the ETF

- **A diversified basket averages the convexity away.** A convertibles ETF blends *balanced*
  converts (convex), *busted* converts (near-pure bond, low equity sensitivity) and
  *equity-like* converts (deep in-the-money, near-linear to the stock). The portfolio
  delta/gamma is a moving average that need not look convex at all — this is the empirical
  question the study settles.
- **The "bond floor" is credit-sensitive and pro-cyclical.** Convertibles are mostly issued
  by lower-rated, growthier firms; the floor is a *risky* bond whose credit spread **widens
  exactly when equities fall** (high-yield credit beta). So the floor drops in the crash
  rather than holding — the opposite of a hedge (cf. the HY-credit literature; the 2008 and
  March-2020 convertible drawdowns).
- **The option is paid for.** Convertibles carry below-market coupons (the issuer sells you
  the conversion option); the embedded option is never free, so the relevant test is whether
  the realised convexity beats a cheap linear stock/bond replica net of fees.

## Method lineage

- **Quadratic / market-timing regression for convexity.** Regressing a security's return on
  the market *and the squared market return* to detect convex (option-like) exposure is the
  Treynor & Mazuy (1966) market-timing test; a positive coefficient on the quadratic term is
  convexity. Henriksson & Merton (1981) is the option-based variant. We use the squared
  *upside* term to isolate upside curvature.
- **Newey–West HAC standard errors** for the convexity coefficient and for the mean of an
  autocorrelated return-difference series: Newey & West (1987), Econometrica.
- **Circular block bootstrap** for CIs on the convexity coefficient and the Sharpe
  *difference* — blocks preserve volatility clustering and cross-asset co-movement that
  i.i.d. resampling destroys (Politis & Romano, 1994; Ledoit & Wolf, *Robust performance
  hypothesis testing with the Sharpe ratio*, JEF 2008).
- **Excess-of-cash comparison.** SHY (1-3y Treasuries) is the cash proxy so the Sharpe race
  between a part-bond instrument (CWB) and a part-bond blend is excess-of-cash vs
  excess-of-cash, per the desk house rule.

## Data sources used

- **CWB** (SPDR Bloomberg Convertible Securities ETF), **SPY** (stocks), **AGG** (iShares
  Core US Aggregate Bond) and **SHY** (1-3y Treasuries, cash proxy), daily,
  **total-return adjusted** (dividends + splits) via `yfinance auto_adjust=True`, cache-first
  through the shared cross-asset panel with a `quantlab.data` fallback per ticker. CWB lists
  **2009-04-16**, which bounds the joint window honestly — stated as a decision, not buried.

## Related desk studies

- [Study 97 — Balancing-Act](../../97-balancing-act/) — the fixed 60/40 stock/bond blend vs
  100% stocks. **This study is distinct**: 97 asks whether *diversification* (a linear
  fixed-weight blend) lifts risk-adjusted return; 339 asks whether a convertibles ETF adds
  *convexity* — a non-linear, asymmetric payoff — beyond such a blend (it doesn't).
- [Study 337 — Covered-Call-ETF](../../337-covered-call-etf/) — the mirror case: covered-call
  ETFs sell a *concave* (capped-upside) payoff as "income"; convertibles claim the *convex*
  side. Both are structured-product wrappers whose marketed payoff shape is the thing under
  test.
