# References & literature map — Study 892 (Corporate-Bond Ladder)

## The claim under test

- **The ladder-vs-fund talking point.** A staple of advisor and personal-finance writing:
  a **bond ladder** (buy bonds maturing in successive years; as each matures, reinvest the
  principal in a new long rung) is said to beat a **constant-maturity bond fund** because
  the ladder holds each bond **to par** and reinvests maturing cash at the *new, higher*
  yield, whereas a fund is "forced to sell" bonds that have fallen in price, "locking in
  losses" — "so a ladder shines through a rate shock like 2022." Circulated widely in
  2022-2024 as the Agg (AGG/BND) posted its worst drawdown on record. Representative
  write-ups: Fidelity, Charles Schwab and Vanguard investor-education pages on "bond ladders
  vs bond funds"; Morningstar's repeated rebuttals ("A bond ladder is not free lunch").

## Why the premium is (mostly) an accounting illusion

- **Held-to-maturity vs mark-to-market for a default-free bond.** The two are the *same*
  economics: a bond bought at par that falls in price after a rate rise **pulls back to par**
  by maturity, and that pull-to-par gain is exactly the reversal of the mark-to-market loss
  the fund reported. Over a full horizon, **two portfolios of equal duration earn the same
  total return** regardless of whether they mark to market or hold to maturity. Standard
  fixed-income texts make the point directly — e.g. **Fabozzi, *Bond Markets, Analysis, and
  Strategies*** (ch. on total return and horizon analysis); **Tuckman & Serrat, *Fixed
  Income Securities*** (the "carry + roll-down + rate change" return decomposition, identical
  for matched-duration books). The only genuine difference between a ladder and a
  constant-maturity fund is the **duration path**: a ladder's duration sawtooths down between
  rolls, averaging roughly half the longest rung, so a ladder is on average a *shorter*
  (lower-risk, lower-return) position than a same-longest-rung fund.
- **Reinvestment risk cuts both ways.** Reinvesting maturing principal at the new yield is an
  advantage only if yields *rose*; a constant-maturity fund is *also* continuously buying new
  bonds at prevailing yields, so the "reinvest at higher yields" benefit is not differential.
  Bierwag/Kaufman/Toevs (immunization literature) formalise that duration-matching, not
  laddering per se, is what pins horizon return.

## The ETF proxy — and why it is imperfect

- **Constant-maturity rungs are not a ladder.** SHY (1-3y), IEI (3-7y), IEF (7-10y) and TLT
  (20y+) are **constant-maturity** iShares Treasury funds — each perpetually rolls to keep
  its band, never letting a bond mature. A fixed-weight basket of them is therefore *another
  constant-maturity portfolio* with a blended duration, **not** a held-to-maturity ladder. A
  true ladder needs **defined-maturity** ETFs — iShares **iBonds** or Invesco **BulletShares**
  — which hold a vintage to a target year and then liquidate (the rung matures). We use the
  SHY/IEI/IEF/TLT mix because it is the liquid, long-history proxy retail investors actually
  reach for; the imperfection is stated on the Signal axis, not hidden.
- **AGG is not pure Treasury.** iShares Core US Aggregate (AGG) and Vanguard Total Bond (BND)
  track the Bloomberg US Aggregate: ~40% Treasury, ~25% MBS, ~25% IG corporate. So the
  Treasury-ladder-vs-AGG gap also carries a **credit/MBS composition** difference — which is
  what drives the 2008 (ladder wins, credit hurt AGG) and 2016-2026 (AGG's spread carry wins)
  era flips in the results.

## Data & durations

- **Prices** — yfinance total-return closes (`auto_adjust=True`) for SHY, IEI, IEF, TLT, AGG,
  BND, LQD (IG-credit cross-check) and BIL (1-3m T-bill cash leg); cached once under
  `_cache/ladder_prices.parquet`. Joint window 2007-06-30 → 2026-06-30 (BND/BIL inception).
- **Effective durations** (years, hardcoded in [`data.py`](../bond_ladder/data.py)) from fund
  fact-sheets (iShares.com / vanguard.com, 2025): SHY 1.9, IEI 4.4, IEF 7.4, TLT 16.5,
  AGG 6.0, BND 5.9, LQD 8.3, BIL 0.1. Used only to duration-match the ladder to the fund;
  the total-return conclusions do not depend on their exact value.

## Method citations

- **Newey & West (1987)**, *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix*, Econometrica — the HAC *t* on the monthly
  ladder-minus-fund difference (6 Bartlett lags).
- **Politis & Romano (1994)**, *The Stationary Bootstrap*, JASA — the circular-block bootstrap
  behind the Sharpe and difference-Sharpe confidence intervals.
- **Lo (2002)**, *The Statistics of Sharpe Ratios*, FAJ — Sharpe-ratio standard errors and the
  autocorrelation caveat (reused via `quantlab.analytics.sharpe_with_se`).

## Related desk studies (dedup)

- [59-downhill](../59-downhill/) — a directional rates/duration timing study; here we do the
  opposite of timing: two static duration-matched structures raced with no view.
- [380-curve-roll-down](../380-curve-roll-down/) — harvests the **roll-down** return along an
  upward-sloping curve. **This study is different**: roll-down accrues to *both* a ladder and
  a constant-maturity fund (both hold aging bonds), so it is not the differential we test —
  we test whether the *ladder wrapper itself* (HTM vs mark-to-market) adds anything.
- [884-convexity-barbell](../884-convexity-barbell/) — a barbell vs bullet **convexity** trade
  (a second-order curvature bet). Our ladder-vs-fund gap is a *first-order* duration + credit
  composition difference, not a convexity harvest.
- [625-starting-yield](../625-starting-yield/) — starting yield as a mechanical predictor of a
  bond fund's forward return (an identity that *does* hold). This study asks the orthogonal
  question — given the same starting yield / duration, does the ladder *structure* beat the
  fund *structure*? — and finds it does not.
