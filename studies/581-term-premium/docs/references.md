# References & literature map — Study 581 (Term-Premium)

## The claim, at full strength

- **Adrian, Crump & Moench (2013)**, *"Pricing the Term Structure with Linear Regressions."*
  *Journal of Financial Economics* 110(1). The **ACM** model — a five-factor affine
  term-structure model estimated by linear regressions — decomposes the long yield into an
  *expectations* component (the average expected future short rate) and a **term premium** (the
  compensation for duration risk). The NY Fed publishes the resulting daily 10-year term-premium
  series; this study builds a retail-reachable *proxy* for it (10-year yield minus an EWMA of the
  short rate) and tests it as a *timing* signal for long-duration returns.
- **Fama & Bliss (1987)**, *"The Information in Long-Maturity Forward Rates."* *American Economic
  Review* 77(4). Forward-spot spreads (a term-premium proxy) predict future bond excess returns —
  the empirical seed of "the term premium times duration."
- **Cochrane & Piazzesi (2005)**, *"Bond Risk Premia."* *American Economic Review* 95(1). A single
  tent-shaped combination of forward rates forecasts one-year excess bond returns with an R² near
  0.35 — strong evidence that the bond risk premium *varies over time* and is *predictable*, the
  premise a term-premium timing rule rests on.
- **Kim & Wright (2005)**, *"An Arbitrage-Free Three-Factor Term Structure Model…"* Federal
  Reserve FEDS 2005-33. The other widely-cited affine term-premium estimate (the "KW" series),
  alongside ACM — both strip the expectations component from the long yield.
- **Ludvigson & Ng (2009)**, *"Macro Factors in Bond Risk Premia."* *Review of Financial Studies*
  22(12). Real and inflation macro factors add to the forward-rate predictors — the term premium
  is partly a compensation that moves with the macro cycle (why the sign is regime-dependent).

## The term-premium estimate we build

- The ACM term premium is ``y10 − 𝔼[average future short rate]``. A no-key retail stack cannot
  reach the affine-model expectations component, so this study proxies it with
  ``tp = y10 − EWMA₍₂₅₂₎(short rate)`` — the 10-year yield (`^TNX`) minus an exponentially-weighted
  moving average of the 3-month bill (`^IRX`). Subtracting the EWMA-expectations term is exactly
  what distinguishes a term-*premium* estimate from the raw 10y−3m *slope*: the slope moves with
  the level and expected path of policy; the premium strips those out. The proxy is a
  simplification, named on the SIGNAL axis, and the reason a `REAL` certification (robust real-tape
  *t* ≥ 2) is out of reach here even in principle.

## Neighbours on this bench (the dedup map)

- **[Study 132 — Yield-Curve-Steepener](../../132-yield-curve-steepener/)** — the *raw* 10y−3m
  curve slope as a TLT timing signal. Study 581 is the **model term-premium** version: it subtracts
  an expectations component (the EWMA of the short rate), which is precisely the piece that turns a
  curve *slope* into a term *premium*. Different signal, same instrument.
- **[Study 380 — Curve-Roll-Down](../../380-curve-roll-down/)** — the carry/roll-down of sitting on
  the curve. Roll-down is the *mechanical* pull-to-par return; the term premium is the *risk
  compensation* embedded in the level of the long yield. Complementary, not the same.
- **[Study 119 — Real-Rate-Regime](../../119-real-rate-regime/)** — conditions equity/bond behaviour
  on the *level* of real rates. Study 581 conditions long-bond returns on the *term premium* (a
  spread/premium, not a level).
- **[Study 247 — Bond-Seasonality](../../247-bond-seasonality/)** — a calendar signal on the same
  TLT instrument; orthogonal to the fundamental term-premium signal here.

## Shared method

- **Newey & West (1987)** — the HAC (heteroskedasticity- and autocorrelation-consistent) *t*-stat
  used on the Q5−Q1 forward-return spread, essential when forward returns overlap.
- **Block / circular-shift permutation testing** (Politis & Romano 1994; Good 2005) — the placebo
  null: rotate the forward-return series in blocks against the term-premium signal and read the
  spread's tail probability, preserving overlap-induced autocorrelation.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (HAC *t* ≥ 2 on
  the real tape for `REAL`; literature support alone reads `WEAK`), one execution lag, costs
  one-way × NAV, and seed-robust synthetic controls (≥ 20 seeds).
